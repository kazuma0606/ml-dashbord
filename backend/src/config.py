"""
環境変数設定モジュール

pydantic-settingsを使用して環境変数から設定を読み込む
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定"""
    
    # データベース設定（個別指定）
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "ml_dashboard"
    
    # データベース設定（URL形式）- RailwayやSupabaseなどで使用
    database_url: Optional[str] = None
    
    # Redis設定（個別指定）
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Redis設定（URL形式）- UpstashやRailwayなどで使用
    redis_url: Optional[str] = None
    
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
    
    def get_database_url(self) -> str:
        """データベース接続URLを取得
        
        DATABASE_URLが設定されている場合はそれを使用、
        なければ個別の設定から構築
        """
        if self.database_url:
            return self.database_url
        
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    def get_redis_url(self) -> str:
        """Redis接続URLを取得
        
        REDIS_URLが設定されている場合はそれを使用、
        なければ個別の設定から構築
        """
        if self.redis_url:
            return self.redis_url
        
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# グローバル設定インスタンス
settings = Settings()

