"""
環境変数設定モジュール

pydantic-settingsを使用して環境変数から設定を読み込む
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定"""
    
    # データベース設定
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "ml_dashboard"
    
    # Redis設定
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # CORS設定
    cors_origins: str = "*"  # カンマ区切りで複数指定可能
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"
    cors_allow_headers: str = "*"
    
    # アプリケーション設定
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# グローバル設定インスタンス
settings = Settings()
