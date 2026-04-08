from fastapi import FastAPI
import pandas as pd
import re
from difflib import get_close_matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

# Download once
nltk.download('stopwords')

app = FastAPI(
    title="Resume Ranking API",
    description="Ranks resumes based on job description",
    version="1.0"
)

# Load data
resume_df = pd.read_csv('Resumes.csv', encoding='latin1')
job_df = pd.read_csv('job_title_des.csv', encoding='latin1')

# Clean column names
job_df.columns = job_df.columns.str.lower().str.strip().str.replace(' ', '_')

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)


@app.get("/")
def home():
    return {"message": "Resume Ranking API is running"}


@app.get("/rank")
def rank_resumes(job_title: str, top_k: int = 10):

    job_title_input = job_title.lower().strip()

    # Find closest matching job title
    job_titles = job_df['job_title'].str.lower().tolist()
    matches = get_close_matches(job_title_input, job_titles, n=1, cutoff=0.3)

    if matches:
        best_match = matches[0]
        filtered_job = job_df[job_df['job_title'].str.lower() == best_match]
    else:
        filtered_job = job_df.iloc[[0]]  # fallback

    job_description = filtered_job.iloc[0]['job_description']

    # Keyword filtering
    keywords = job_title_input.split()

    filtered_resumes = resume_df[
        resume_df['Resume_str'].str.lower().apply(
            lambda x: any(k in x for k in keywords)
        )
    ]

    # fallback if no resumes matched
    if filtered_resumes.empty:
        filtered_resumes = resume_df

    filtered_resumes = filtered_resumes.copy()

    # Clean text
    filtered_resumes['clean_resume'] = filtered_resumes['Resume_str'].apply(clean_text)
    clean_job_description = clean_text(job_description)

    # TF-IDF
    tfidf = TfidfVectorizer()

    all_text = filtered_resumes['clean_resume'].tolist()
    all_text.append(clean_job_description)

    tfidf_matrix = tfidf.fit_transform(all_text)

    # Similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[:-1],
        tfidf_matrix[-1:]
    ).flatten()

    filtered_resumes['similarity_score'] = similarity_scores.round(3)

    # Sort
    ranked_resumes = filtered_resumes.sort_values(
        by='similarity_score',
        ascending=False
    )

    # Preview
    ranked_resumes['resume_preview'] = ranked_resumes['Resume_str'].str[:200]

    result = ranked_resumes[['resume_preview', 'similarity_score']].head(top_k)

    return result.to_dict(orient="records")