"""
Step 3 of the pipeline: text -> vector.
Uses Google's Gemini Embedding API (hosted, free tier, no credit card
required) instead of running any model in-process. This is the key fix
for Render's 512MB free-tier memory limit - no ML model ever loads into
this container's RAM, since the embedding computation happens on
Google's servers via a plain HTTP call.
"""
from typing import List
from google import genai
from app.config import settings

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def embed_text(text: str) -> List[float]:
    result = get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


def embed_batch(texts: List[str]) -> List[List[float]]:
    result = get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in result.embeddings]