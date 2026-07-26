"""
TruthCheck — Misinformation Detector
Single-file backend: translation + similarity search + Gemini reasoning, all in one.
"""

import os
import json
import pandas as pd
import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import google.generativeai as genai

# ---------- SETUP ----------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.0-flash"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "factchecks.csv")

app = FastAPI(title="Misinformation Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CheckRequest(BaseModel):
    text: str


# ---------- LOAD EMBEDDING MODEL + FACT-CHECK DATASET ----------

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_df = pd.read_csv(DATA_PATH)
_embeddings = _embed_model.encode(_df["claim"].tolist(), convert_to_numpy=True)
faiss.normalize_L2(_embeddings)
_index = faiss.IndexFlatIP(_embeddings.shape[1])
_index.add(_embeddings)


# ---------- HELPERS ----------

def translate_to_english(text: str):
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang == "en":
        return text, lang

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated, lang
    except Exception as e:
        print(f"Translation failed: {e}")
        return text, lang


def find_similar_claims(query: str, top_k: int = 3):
    query_emb = _embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    scores, indices = _index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = _df.iloc[idx]
        results.append({
            "claim": str(row["claim"]),
            "verdict": str(row["verdict"]),
            "source": str(row["source"]),
            "url": str(row["url"]),
            "similarity": float(score),
        })
    return results


def analyze_claim(text: str, similar_claims: list):
    context_block = ""
    if similar_claims:
        context_block = "Known fact-checked claims that may be related:\n"
        for c in similar_claims:
            context_block += (
                f"- Claim: \"{c['claim']}\" | Verdict: {c['verdict']} "
                f"| Source: {c['source']} | Similarity: {c['similarity']:.2f}\n"
            )

    prompt = f"""You are a misinformation-detection assistant. Analyze the following
message and respond ONLY with valid JSON, no preamble, no markdown formatting, no code fences.

Message to analyze:
\"\"\"{text}\"\"\"

{context_block}

Return JSON in exactly this shape:
{{
  "extracted_claim": "the core factual claim being made, in one sentence",
  "verdict": "Likely True" or "Unverified" or "Likely False",
  "confidence": a number from 0 to 100,
  "explanation": "2-3 sentence plain-language explanation of the reasoning, mention if it matches a known fact-checked claim"
}}
"""

    models_to_try = [GEMINI_MODEL, "gemini-2.0-flash-lite", "gemini-flash-latest"]
    result = None

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw_text)
            break
        except Exception as e:
            print(f"Gemini model {model_name} error: {e}")

    if not result:
        # Fallback to similarity search top match if available
        top_match = similar_claims[0] if (similar_claims and similar_claims[0]["similarity"] >= 0.50) else None
        if top_match:
            v_upper = str(top_match["verdict"]).upper()
            if "FALSE" in v_upper:
                verdict = "Likely False"
            elif "TRUE" in v_upper:
                verdict = "Likely True"
            else:
                verdict = "Unverified"
            conf = int(min(98, max(65, top_match["similarity"] * 100)))
            result = {
                "extracted_claim": top_match["claim"],
                "verdict": verdict,
                "confidence": conf,
                "explanation": f"Matched known fact-check by {top_match['source']}: '{top_match['claim']}' (Similarity: {top_match['similarity']*100:.0f}%).",
            }
        else:
            result = {
                "extracted_claim": text,
                "verdict": "Unverified",
                "confidence": 50,
                "explanation": "Could not confidently analyze this claim. Please verify manually with a trusted fact-checking source.",
            }

    return result


# ---------- ROUTES ----------

@app.get("/")
def root():
    return {"status": "ok", "message": "Misinformation Detector API is running"}


@app.post("/check")
def check_claim(req: CheckRequest):
    original_text = req.text
    translated_text, detected_lang = translate_to_english(original_text)
    similar = find_similar_claims(translated_text, top_k=3)
    analysis = analyze_claim(translated_text, similar)

    return {
        "original_text": original_text,
        "detected_language": detected_lang,
        "translated_text": translated_text,
        "similar_claims": similar,
        "extracted_claim": analysis["extracted_claim"],
        "verdict": analysis["verdict"],
        "confidence": analysis["confidence"],
        "explanation": analysis["explanation"],
    }
