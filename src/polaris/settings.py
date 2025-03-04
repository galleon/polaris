from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: str
    # GOOGLE_GENAI_API_KEY: str
    AOC_SESSION_TOKEN: str
    HF_TOKEN: str
    QDRANT_CLOUD: str
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_INDEX_NAME: str
    QDRANT_TOKEN: str

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
