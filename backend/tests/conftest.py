"""
Pytestフィクスチャ設定
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from testcontainers.postgres import PostgresContainer

from src.models.database import Base


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQLテストコンテナ"""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


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
