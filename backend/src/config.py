"""
環境変数設定モジュール

pydantic-settingsを使用して環境変数から設定を読み込む
"""
import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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
        # 環境変数から直接取得も試みる（pydantic-settingsが読み込めない場合のフォールバック）
        database_url = self.database_url or os.getenv("DATABASE_URL")
        
        if database_url:
            logger.info(f"DATABASE_URLを使用: {database_url.split('@')[0] if '@' in database_url else 'set'}@***")
            return database_url
        
        url = (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
        logger.info(f"個別設定からデータベースURLを構築: postgresql://{self.postgres_user}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}")
        return url
    
    def get_redis_url(self) -> str:
        """Redis接続URLを取得
        
        REDIS_URLが設定されている場合はそれを使用、
        なければ個別の設定から構築
        """
        # 環境変数から直接取得も試みる
        redis_url = self.redis_url or os.getenv("REDIS_URL")
        
        if redis_url:
            logger.info(f"REDIS_URLを使用: {redis_url.split('@')[0] if '@' in redis_url else 'set'}@***")
            return redis_url
        
        url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
        logger.info(f"個別設定からRedis URLを構築: redis://{self.redis_host}:{self.redis_port}/{self.redis_db}")
        return url


# グローバル設定インスタンス
settings = Settings()

# 起動時に設定をログ出力
logger.info("=== 環境変数設定 ===")
logger.info(f"DATABASE_URL環境変数: {'設定あり' if os.getenv('DATABASE_URL') else '設定なし'}")
logger.info(f"REDIS_URL環境変数: {'設定あり' if os.getenv('REDIS_URL') else '設定なし'}")
logger.info(f"POSTGRES_HOST: {settings.postgres_host}")
logger.info(f"REDIS_HOST: {settings.redis_host}")

