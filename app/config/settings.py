import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database configurations
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "bookdb")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    # Database URL
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
    CSV_PATH: str = os.getenv("CSV_PATH", f"{DATA_DIR}/data.csv")
    EMBEDDINGS_PATH: str = os.getenv("EMBEDDINGS_PATH", f"{DATA_DIR}/embeddings.npy")
    
    # API settings
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Vector search settings
    VECTOR_DIMENSION: int = 384  # Assuming the dimension of your embeddings
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    DEFAULT_MAX_RESULTS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()