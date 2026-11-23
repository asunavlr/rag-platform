from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Platform"
    environment: str = "development"
    database_url: str = "sqlite:///./rag_platform.db"
    jwt_secret: str = "development-only-change-me"
    jwt_expire_minutes: int = Field(default=1440, ge=5, le=10080)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    embedding_provider: str = "local"
    embedding_dimensions: int = Field(default=1536, ge=8, le=3072)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: int = Field(default=45, ge=5, le=180)

    @model_validator(mode="after")
    def validate_production_secrets(self):
        if self.environment.lower() == "production":
            if self.jwt_secret == "development-only-change-me" or len(self.jwt_secret) < 32:
                raise ValueError("Set a unique JWT_SECRET with at least 32 characters in production")
            if self.embedding_provider == "openai" and not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return self


settings = Settings()
