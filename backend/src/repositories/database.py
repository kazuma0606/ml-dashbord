"""
データベース接続とセッション管理
"""
import time
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from ..config import settings
from ..models.database import Base
from ..models.exceptions import DatabaseError

logger = logging.getLogger(__name__)

# データベースURL構築
DATABASE_URL = (
    f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
)

# エンジン作成
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# セッションファクトリー
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(max_retries: int = 5) -> None:
    """
    データベース初期化（テーブル作成）
    
    Args:
        max_retries: 最大リトライ回数
        
    Raises:
        DatabaseError: データベース接続に失敗した場合
    """
    retry_delays = [1, 2, 4, 8, 16]  # 指数バックオフ
    
    for attempt in range(max_retries):
        try:
            logger.info(f"データベース接続試行 {attempt + 1}/{max_retries}")
            Base.metadata.create_all(bind=engine)
            logger.info("データベース初期化成功")
            return
        except OperationalError as e:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logger.warning(
                    f"データベース接続失敗（試行 {attempt + 1}/{max_retries}）: {e}. "
                    f"{delay}秒後にリトライします..."
                )
                time.sleep(delay)
            else:
                logger.error(f"データベース接続失敗（最大リトライ回数到達）: {e}")
                raise DatabaseError(f"データベース接続に失敗しました: {e}")


def get_db() -> Generator[Session, None, None]:
    """
    データベースセッション取得（依存性注入用）
    
    Yields:
        Session: SQLAlchemyセッション
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
