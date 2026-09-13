"""
Step 3 of the pipeline: text -> vector.
Uses Google's Gemini Embedding API (hosted, free tier, no credit card
required) instead of running any model in-process.

gemini-embedding-001 returns 3072-dimension vectors by default - we
explicitly request a smaller size via output_dimensionality to match
our database schema and keep storage smaller.
"""
from typing import List
from google import genai
from google.genai import types
from app.config import settings

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _embed_config() -> types.EmbedContentConfig:
    return types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIM)


def embed_text(text: str) -> List[float]:
    result = get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=_embed_config(),
    )
    return result.embeddings[0].values


def embed_batch(texts: List[str]) -> List[List[float]]:
    result = get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=_embed_config(),
    )
    return [e.values for e in result.embeddings]