from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Platform"
    environment: str = "development"
    database_url: str = "sqlite:///./rag_platform.db"
    jwt_secret: str = "development-only-change-me"
    jwt_expire_minutes: int = 1440
    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 900
    chunk_overlap: int = 150
    embedding_provider: str = "local"
    embedding_dimensions: int = 256
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: int = 45


settings = Settings()
