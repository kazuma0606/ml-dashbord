"""Frontend configuration using environment variables."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Frontend application settings."""
    
    # Backend API Configuration
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = 30
    
    # Application Settings
    app_title: str = "ML Dashboard"
    app_icon: str = "🤖"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
