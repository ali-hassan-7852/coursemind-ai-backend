"""
Central place for all environment/config values.
Every other file should import `settings` from here instead of
reading os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/coursemind"
    )

    # Auth / JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-this-secret-key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # Embeddings (Gemini hosted API - see app/services/embedding.py)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # LLM (answer generation)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")


settings = Settings()