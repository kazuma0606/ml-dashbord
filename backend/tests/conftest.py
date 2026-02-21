"""
Pytestフィクスチャ設定
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from src.models.database import Base
from src.services.cache import CacheManager


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQLテストコンテナ"""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def redis_container():
    """Redisテストコンテナ"""
    with RedisContainer("redis:7") as redis:
        yield redis


@pytest.fixture(scope="session")
def test_cache_manager(redis_container):
    """テスト用キャッシュマネージャー"""
    # Redisコンテナの接続情報を取得
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    
    cache_manager = CacheManager(host=host, port=int(port), db=0)
    yield cache_manager
    cache_manager.close()


@pytest.fixture(scope="session")
def test_engine(postgres_container):
    """テスト用データベースエンジン"""
    engine = create_engine(postgres_container.get_connection_url())
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine) -> Session:
    """テスト用データベースセッション（各テストで独立）"""
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        
        # テーブルをクリーンアップ
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
