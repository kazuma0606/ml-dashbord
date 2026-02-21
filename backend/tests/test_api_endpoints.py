"""
APIエンドポイントのテスト
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from backend.src.main import app
from backend.src.repositories.database import Base, get_db
from backend.src.models.database import Experiment


# テスト用インメモリデータベース
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# テーブルを作成
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    """テスト用データベースセッション"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# FastAPIの依存関係をオーバーライド
app.dependency_overrides[get_db] = override_get_db

# テストクライアント（lifespanを無効化）
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def setup_database():
    """各テスト後にデータベースをクリーンアップ"""
    yield
    # テスト後にデータをクリア
    db = TestingSessionLocal()
    try:
        db.query(Experiment).delete()
        db.commit()
    finally:
        db.close()


def test_health_check():
    """ヘルスチェックエンドポイントのテスト"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_get_datasets():
    """データセット一覧取得エンドポイントのテスト"""
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) > 0
    assert all("name" in ds and "description" in ds for ds in data["datasets"])


def test_get_dataset_metadata():
    """データセットメタデータ取得エンドポイントのテスト"""
    response = client.get("/api/datasets/iris")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "n_samples" in data
    assert "n_features" in data
    assert data["name"] == "iris"


def test_get_dataset_not_found():
    """存在しないデータセットのテスト"""
    response = client.get("/api/datasets/nonexistent")
    assert response.status_code == 404


def test_get_dataset_preview():
    """データセットプレビュー取得エンドポイントのテスト"""
    response = client.get("/api/datasets/iris/preview?n_rows=5")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "columns" in data
    assert "n_rows" in data
    assert data["n_rows"] <= 5


def test_train_model():
    """モデル学習エンドポイントのテスト"""
    training_config = {
        "dataset_name": "iris",
        "test_size": 0.3,
        "random_state": 42,
        "model_type": "random_forest",
        "hyperparameters": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42
        }
    }
    
    response = client.post("/api/train", json=training_config)
    assert response.status_code == 200
    data = response.json()
    
    # 必須フィールドの確認
    assert "model_id" in data
    assert "accuracy" in data
    assert "f1_score" in data
    assert "confusion_matrix" in data
    assert "classification_report" in data
    assert "training_time" in data
    
    # 値の範囲確認
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["f1_score"] <= 1.0
    assert data["training_time"] > 0


def test_get_model_info():
    """モデル情報取得エンドポイントのテスト"""
    # まずモデルを学習
    training_config = {
        "dataset_name": "iris",
        "test_size": 0.3,
        "random_state": 42,
        "model_type": "logistic_regression",
        "hyperparameters": {"random_state": 42}
    }
    
    train_response = client.post("/api/train", json=training_config)
    model_id = train_response.json()["model_id"]
    
    # モデル情報を取得
    response = client.get(f"/api/models/{model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == model_id
    assert data["dataset_name"] == "iris"
    assert data["model_type"] == "logistic_regression"


def test_export_model():
    """モデルエクスポートエンドポイントのテスト"""
    # まずモデルを学習
    training_config = {
        "dataset_name": "iris",
        "test_size": 0.3,
        "random_state": 42,
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 5, "random_state": 42}
    }
    
    train_response = client.post("/api/train", json=training_config)
    model_id = train_response.json()["model_id"]
    
    # モデルをエクスポート
    response = client.get(f"/api/models/{model_id}/export")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "Content-Disposition" in response.headers
    assert len(response.content) > 0


def test_save_experiment():
    """実験記録保存エンドポイントのテスト
    
    Note: このテストはPostgreSQLデータベースが必要なため、
    統合テスト環境でのみ実行されます
    """
    pytest.skip("Requires PostgreSQL database - run in integration test environment")
    experiment = {
        "dataset_name": "iris",
        "model_type": "random_forest",
        "accuracy": 0.95,
        "f1_score": 0.94,
        "hyperparameters": {"n_estimators": 100},
        "training_time": 1.5
    }
    
    response = client.post("/api/experiments", json=experiment)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "message" in data


def test_get_experiments():
    """実験履歴取得エンドポイントのテスト
    
    Note: このテストはPostgreSQLデータベースが必要なため、
    統合テスト環境でのみ実行されます
    """
    pytest.skip("Requires PostgreSQL database - run in integration test environment")
    # まず実験を保存
    experiment = {
        "dataset_name": "iris",
        "model_type": "random_forest",
        "accuracy": 0.95,
        "f1_score": 0.94,
        "hyperparameters": {"n_estimators": 100},
        "training_time": 1.5
    }
    client.post("/api/experiments", json=experiment)
    
    # 実験履歴を取得
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert len(data["experiments"]) > 0


def test_clear_experiments():
    """実験履歴クリアエンドポイントのテスト
    
    Note: このテストはPostgreSQLデータベースが必要なため、
    統合テスト環境でのみ実行されます
    """
    pytest.skip("Requires PostgreSQL database - run in integration test environment")
    # まず実験を保存
    experiment = {
        "dataset_name": "iris",
        "model_type": "random_forest",
        "accuracy": 0.95,
        "f1_score": 0.94,
        "hyperparameters": {"n_estimators": 100},
        "training_time": 1.5
    }
    client.post("/api/experiments", json=experiment)
    
    # 実験履歴をクリア
    response = client.delete("/api/experiments")
    assert response.status_code == 200
    
    # クリアされたことを確認
    get_response = client.get("/api/experiments")
    assert len(get_response.json()["experiments"]) == 0
