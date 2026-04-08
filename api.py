from fastapi import FastAPI
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = FastAPI()

# Load data
resume_df = pd.read_csv('Resumes.csv', encoding='latin1')  
job_df = pd.read_csv('job_title_des.csv', encoding='latin1')

# Clean column names
job_df.columns = job_df.columns.str.lower().str.strip().str.replace(' ', '_')

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Home API
@app.get("/")
def home():
    return {"message": "Resume Ranking API is running"}

# Rank API
@app.get("/rank")
def rank_resumes(job_title: str, top_k: int = 10):

    # normalize input
    job_title_input = job_title.lower().strip()

    synonyms = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "ds": "data scientist",
        "hr": "human resources"
    }

    if job_title_input in synonyms:
        job_title_input = synonyms[job_title_input]

    filtered_job = job_df[
        job_df['job_title'].str.lower().str.contains(job_title_input, na=False)
    ]

    if filtered_job.empty:
        return {"error": "Job title not found"}

    job_description = filtered_job.iloc[0]['job_description']

    resume_df['clean_resume'] = resume_df['Resume_str'].apply(clean_text)
    clean_job_description = clean_text(job_description)

    tfidf = TfidfVectorizer()

    all_text = resume_df['clean_resume'].tolist()
    all_text.append(clean_job_description)

    tfidf_matrix = tfidf.fit_transform(all_text)

    similarity_scores = cosine_similarity(
        tfidf_matrix[:-1],
        tfidf_matrix[-1:]
    ).flatten()

    resume_df['similarity_score'] = similarity_scores.round(3)

    ranked_resumes = resume_df.sort_values(
        by='similarity_score',
        ascending=False
    )

    ranked_resumes['resume_preview'] = ranked_resumes['Resume_str'].str[:200]

    result = ranked_resumes[['resume_preview', 'similarity_score']].head(top_k)

    return result.to_dict(orient="records")