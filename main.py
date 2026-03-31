import joblib
import pdfplumber
import numpy as np
import io
import re

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

skill_model = joblib.load("skills_model.pkl")
mlb = joblib.load("mlb.pkl")
exp_model = joblib.load("exp_model.pkl")
vectorizer_exp = joblib.load("exp_vectorizer.pkl")

skills_list = list(mlb.classes_)

app = FastAPI(title="Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        raise HTTPException(status_code=400, detail="Error reading PDF")

    return text.lower()


def extract_all_skills(text, skills_list):
    text = text.lower()
    found = []

    for skill in skills_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            found.append(skill)

    return list(set(found))


def predict_resume(text):

    pred_skills = skill_model.predict([text])
    ml_skills = mlb.inverse_transform(pred_skills)[0]

    all_skills = extract_all_skills(text, skills_list)

    final_skills = list(set(ml_skills) | set(all_skills))

 
    vec = vectorizer_exp.transform([text])
    exp = exp_model.predict(vec)[0]

    return {
        "skills": sorted(final_skills),
        "experience_years": float(round(exp, 2))
    }


@app.post("/analyze_resume")
async def analyze_resume(file: UploadFile = File(...)):


    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")


    contents = await file.read()


    text = extract_text_from_pdf(io.BytesIO(contents))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF")

 
    result = predict_resume(text)

    return {
        "filename": file.filename,
        "result": result
    }


@app.post("/analyze_text")
def analyze_text(text: str):
    return predict_resume(text)