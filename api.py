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

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Home route
@app.get("/")
def home():
    return {"message": "Resume Ranking API is running"}

# Rank route
@app.get("/rank")
def rank_resumes(job_title: str, top_k: int = 10):

    job_title_input = job_title.lower().strip()

    # find matching job (case-insensitive + partial match)
    filtered_job = job_df[
        job_df['job_title'].str.lower().str.contains(job_title_input, na=False)
    ]

    if filtered_job.empty:
        return {"error": "Job title not found"}

    job_description = filtered_job.iloc[0]['job_description']

    # combine title + description (important for accuracy)
    combined_text = job_title_input + " " + str(job_description)

    # clean text
    resume_df['clean_resume'] = resume_df['Resume_str'].apply(clean_text)
    clean_job_description = clean_text(combined_text)

    # TF-IDF (improved with bigrams)
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)

    all_text = resume_df['clean_resume'].tolist()
    all_text.append(clean_job_description)

    tfidf_matrix = tfidf.fit_transform(all_text)

    # similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[:-1],
        tfidf_matrix[-1:]
    ).flatten()

    # keyword boost (simple + effective)
    keywords = job_title_input.split()

    def keyword_score(text):
        return sum(1 for k in keywords if k in text)

    resume_df['keyword_score'] = resume_df['clean_resume'].apply(keyword_score)

    # final score
    resume_df['final_score'] = similarity_scores + (resume_df['keyword_score'] * 0.1)

    # ranking
    ranked_resumes = resume_df.sort_values(
        by='final_score',
        ascending=False
    )

    # preview text
    ranked_resumes['resume_preview'] = ranked_resumes['Resume_str'].str[:200]

    result = ranked_resumes[['resume_preview', 'final_score']].head(top_k)

    return result.to_dict(orient="records")