from fastapi import FastAPI
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

# download once
nltk.download('stopwords')

app = FastAPI()

# Load data
resume_df = pd.read_csv('Resumes.csv', encoding='latin1')
job_df = pd.read_csv('job_title_des.csv', encoding='latin1')

# Clean column names
job_df.columns = job_df.columns.str.lower().str.strip().str.replace(' ', '_')

# Stopwords
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Home
@app.get("/")
def home():
    return {"message": "Resume Ranking API is running"}

# Rank API
@app.get("/rank")
def rank_resumes(job_title: str, top_k: int = 10):

    job_title_input = job_title.lower().strip()

    filtered_job = job_df[
        job_df['job_title'].str.lower().str.contains(job_title_input, na=False)
    ]

    if filtered_job.empty:
        return {"error": "Job title not found"}

    job_description = filtered_job.iloc[0]['job_description']

    combined_text = job_title_input + " " + str(job_description)

    keywords = job_title_input.split()

    filtered_resumes = resume_df[
        resume_df['Resume_str'].str.lower().apply(
            lambda x: any(k in x for k in keywords)
        )
    ]

    # fallback if no match
    if filtered_resumes.empty:
        filtered_resumes = resume_df.copy()

    filtered_resumes['clean_resume'] = filtered_resumes['Resume_str'].apply(clean_text)
    clean_job_description = clean_text(combined_text)

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)

    all_text = filtered_resumes['clean_resume'].tolist()
    all_text.append(clean_job_description)

    tfidf_matrix = tfidf.fit_transform(all_text)

    similarity_scores = cosine_similarity(
        tfidf_matrix[:-1],
        tfidf_matrix[-1:]
    ).flatten()

    def keyword_score(text):
        return sum(1 for k in keywords if k in text)

    filtered_resumes['keyword_score'] = filtered_resumes['clean_resume'].apply(keyword_score)

    filtered_resumes['final_score'] = similarity_scores + (filtered_resumes['keyword_score'] * 0.1)

    ranked_resumes = filtered_resumes.sort_values(
        by='final_score',
        ascending=False
    )

    ranked_resumes['resume_preview'] = ranked_resumes['Resume_str'].str[:200]

    result = ranked_resumes[['resume_preview', 'final_score']].head(top_k)

    return result.to_dict(orient="records")