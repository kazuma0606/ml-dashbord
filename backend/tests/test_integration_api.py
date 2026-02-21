"""
統合テスト: APIエンドポイント

testcontainersを使用してPostgreSQLとRedisコンテナを起動し、
すべてのAPIエンドポイントを実際のデータベース・キャッシュ環境でテストする。

要件: 9.1, 9.2, 9.3
"""
import pytest
import pickle
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from src.main import app
from src.models.database import Base
from src.repositories.database import get_db
from src.services.cache import CacheManager, get_cache_manager


@pytest.fixture(scope="module")
def postgres_container():
    """PostgreSQLテストコンテナ（モジュールスコープ）"""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture(scope="module")
def redis_container():
    """Redisテストコンテナ（モジュールスコープ）"""
    with RedisContainer("redis:7") as redis:
        yield redis


@pytest.fixture(scope="module")
def test_engine(postgres_container):
    """テスト用データベースエンジン"""
    engine = create_engine(postgres_container.get_connection_url())
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def test_cache_manager(redis_container):
    """テスト用キャッシュマネージャー"""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    
    cache_manager = CacheManager(host=host, port=int(port), db=0)
    yield cache_manager
    # クリーンアップ
    try:
        cache_manager.client.flushdb()
    except:
        pass
    cache_manager.close()


@pytest.fixture(scope="module")
def test_client(test_engine, test_cache_manager):
    """統合テスト用FastAPIクライアント"""
    # グローバルキャッシュマネージャーをリセット
    import src.services.cache as cache_module
    cache_module._cache_manager = test_cache_manager
    
    # データベース依存関係をオーバーライド
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    # キャッシュマネージャーをオーバーライド
    def override_get_cache_manager():
        return test_cache_manager
    
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_cache_manager] = override_get_cache_manager
    
    # lifespanを無効化してテストクライアントを作成
    client = TestClient(app, raise_server_exceptions=False)
    
    yield client
    
    # クリーンアップ
    app.dependency_overrides.clear()
    cache_module._cache_manager = None


@pytest.fixture(autouse=True)
def cleanup_database(test_engine):
    """各テスト後にデータベースをクリーンアップ"""
    yield
    # テスト後にデータをクリア
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSessionLocal()
    try:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    finally:
        session.close()


class TestHealthCheck:
    """ヘルスチェックエンドポイントのテスト"""
    
    def test_health_check(self, test_client):
        """ヘルスチェックが正常に動作すること"""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestDatasetEndpoints:
    """データセット関連エンドポイントの統合テスト"""
    
    def test_get_datasets(self, test_client):
        """データセット一覧取得が正常に動作すること
        
        要件: 1.1
        """
        response = test_client.get("/api/datasets")
        assert response.status_code == 200
        
        data = response.json()
        assert "datasets" in data
        assert len(data["datasets"]) > 0
        
        # 各データセットに必要な情報が含まれていること
        for dataset in data["datasets"]:
            assert "name" in dataset
            assert "description" in dataset
    
    def test_get_dataset_metadata(self, test_client):
        """データセットメタデータ取得が正常に動作すること
        
        要件: 1.1, 1.5
        """
        response = test_client.get("/api/datasets/iris")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "iris"
        assert "n_samples" in data
        assert "n_features" in data
        assert data["n_samples"] > 0
        assert data["n_features"] > 0
    
    def test_get_dataset_not_found(self, test_client):
        """存在しないデータセットで404エラーが返されること"""
        response = test_client.get("/api/datasets/nonexistent_dataset")
        assert response.status_code == 404
    
    def test_get_dataset_preview(self, test_client):
        """データセットプレビュー取得が正常に動作すること
        
        要件: 1.4
        """
        n_rows = 5
        response = test_client.get(f"/api/datasets/iris/preview?n_rows={n_rows}")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert "columns" in data
        assert "n_rows" in data
        assert data["n_rows"] <= n_rows
        assert len(data["columns"]) > 0
    
    def test_dataset_caching(self, test_client, test_cache_manager):
        """データセットがRedisにキャッシュされること
        
        要件: 7.1, 7.2
        """
        # キャッシュをクリア
        test_cache_manager.client.flushdb()
        
        # プレビューエンドポイントを使用（これは確実にload_datasetを呼ぶ）
        response1 = test_client.get("/api/datasets/iris/preview?n_rows=5")
        assert response1.status_code == 200
        
        # キャッシュキーが存在することを確認
        cache_key = "dataset:iris"
        # Redisに直接問い合わせ
        exists = test_cache_manager.client.exists(cache_key)
        assert exists > 0, f"Dataset should be cached after first request. Keys in cache: {test_cache_manager.client.keys('*')}"
        
        # 2回目のリクエスト（キャッシュから取得）
        response2 = test_client.get("/api/datasets/iris/preview?n_rows=5")
        assert response2.status_code == 200
        
        # 同じデータが返されること
        assert response1.json() == response2.json()


class TestModelTrainingEndpoints:
    """モデル学習関連エンドポイントの統合テスト"""
    
    def test_train_model_random_forest(self, test_client):
        """Random Forestモデルの学習が正常に動作すること
        
        要件: 3.1, 4.1, 4.2, 4.4, 4.5, 4.6
        """
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
        
        response = test_client.post("/api/train", json=training_config)
        assert response.status_code == 200
        
        data = response.json()
        
        # 必須フィールドの確認
        assert "model_id" in data
        assert "accuracy" in data
        assert "f1_score" in data
        assert "confusion_matrix" in data
        assert "classification_report" in data
        assert "training_time" in data
        assert "feature_importances" in data  # Random Forestは特徴量重要度を持つ
        
        # 値の範囲確認
        assert 0.0 <= data["accuracy"] <= 1.0
        assert 0.0 <= data["f1_score"] <= 1.0
        assert data["training_time"] > 0
        
        # 混同行列の形状確認
        assert len(data["confusion_matrix"]) > 0
        assert len(data["confusion_matrix"][0]) > 0
        
        # 特徴量重要度が降順であること（プロパティ10）
        if data["feature_importances"]:
            importances = data["feature_importances"]
            assert importances == sorted(importances, reverse=True)
    
    def test_train_model_logistic_regression(self, test_client):
        """Logistic Regressionモデルの学習が正常に動作すること
        
        要件: 3.1
        """
        training_config = {
            "dataset_name": "iris",
            "test_size": 0.3,
            "random_state": 42,
            "model_type": "logistic_regression",
            "hyperparameters": {
                "C": 1.0,
                "random_state": 42
            }
        }
        
        response = test_client.post("/api/train", json=training_config)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_id" in data
        assert 0.0 <= data["accuracy"] <= 1.0
        assert data["feature_importances"] is None  # Logistic Regressionは特徴量重要度を持たない
    
    def test_train_model_with_invalid_dataset(self, test_client):
        """無効なデータセット名でエラーが返されること
        
        要件: 3.4
        """
        training_config = {
            "dataset_name": "invalid_dataset",
            "test_size": 0.3,
            "random_state": 42,
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10}
        }
        
        response = test_client.post("/api/train", json=training_config)
        assert response.status_code == 404
    
    def test_get_model_info(self, test_client):
        """モデル情報取得が正常に動作すること"""
        # まずモデルを学習
        training_config = {
            "dataset_name": "iris",
            "test_size": 0.3,
            "random_state": 42,
            "model_type": "logistic_regression",
            "hyperparameters": {"random_state": 42}
        }
        
        train_response = test_client.post("/api/train", json=training_config)
        model_id = train_response.json()["model_id"]
        
        # モデル情報を取得
        response = test_client.get(f"/api/models/{model_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_id"] == model_id
        assert data["dataset_name"] == "iris"
        assert data["model_type"] == "logistic_regression"
        assert "config" in data
    
    def test_get_model_info_not_found(self, test_client):
        """存在しないモデルIDで404エラーが返されること"""
        response = test_client.get("/api/models/nonexistent_model_id")
        assert response.status_code == 404
    
    def test_export_model(self, test_client):
        """モデルエクスポートが正常に動作すること
        
        要件: 6.1, 6.2
        """
        # まずモデルを学習
        training_config = {
            "dataset_name": "iris",
            "test_size": 0.3,
            "random_state": 42,
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 5, "random_state": 42}
        }
        
        train_response = test_client.post("/api/train", json=training_config)
        model_id = train_response.json()["model_id"]
        
        # モデルをエクスポート
        response = test_client.get(f"/api/models/{model_id}/export")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert "Content-Disposition" in response.headers
        assert len(response.content) > 0
        
        # pickleファイルとしてデシリアライズ可能であること
        try:
            model = pickle.loads(response.content)
            assert model is not None
        except Exception as e:
            pytest.fail(f"Failed to deserialize model: {e}")


class TestExperimentEndpoints:
    """実験履歴関連エンドポイントの統合テスト（PostgreSQL使用）"""
    
    def test_save_experiment(self, test_client):
        """実験記録の保存が正常に動作すること
        
        要件: 5.1, 5.2
        """
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": "iris",
            "model_type": "random_forest",
            "accuracy": 0.95,
            "f1_score": 0.94,
            "hyperparameters": {"n_estimators": 100, "max_depth": 5},
            "training_time": 1.5
        }
        
        response = test_client.post("/api/experiments", json=experiment)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "message" in data
        assert data["id"] > 0
    
    def test_get_experiments(self, test_client):
        """実験履歴取得が正常に動作すること
        
        要件: 5.2, 5.3
        """
        # まず複数の実験を保存
        experiments = [
            {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": "iris",
                "model_type": "random_forest",
                "accuracy": 0.95,
                "f1_score": 0.94,
                "hyperparameters": {"n_estimators": 100},
                "training_time": 1.5
            },
            {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": "wine",
                "model_type": "logistic_regression",
                "accuracy": 0.92,
                "f1_score": 0.91,
                "hyperparameters": {"C": 1.0},
                "training_time": 0.8
            }
        ]
        
        for exp in experiments:
            test_client.post("/api/experiments", json=exp)
        
        # 実験履歴を取得
        response = test_client.get("/api/experiments")
        assert response.status_code == 200
        
        data = response.json()
        assert "experiments" in data
        assert len(data["experiments"]) >= 2
        
        # 各実験記録に必要なフィールドが含まれていること
        for exp in data["experiments"]:
            assert "id" in exp
            assert "timestamp" in exp
            assert "dataset_name" in exp
            assert "model_type" in exp
            assert "accuracy" in exp
            assert "f1_score" in exp
            assert "hyperparameters" in exp
            assert "training_time" in exp
        
        # 時系列降順でソートされていること（プロパティ13）
        timestamps = [exp["timestamp"] for exp in data["experiments"]]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_clear_experiments(self, test_client):
        """実験履歴クリアが正常に動作すること
        
        要件: 5.5
        """
        # まず実験を保存
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": "iris",
            "model_type": "random_forest",
            "accuracy": 0.95,
            "f1_score": 0.94,
            "hyperparameters": {"n_estimators": 100},
            "training_time": 1.5
        }
        test_client.post("/api/experiments", json=experiment)
        
        # 実験履歴をクリア
        response = test_client.delete("/api/experiments")
        assert response.status_code == 200
        assert "message" in response.json()
        
        # クリアされたことを確認
        get_response = test_client.get("/api/experiments")
        assert get_response.status_code == 200
        assert len(get_response.json()["experiments"]) == 0
    
    def test_experiment_roundtrip(self, test_client):
        """実験記録の保存と取得のラウンドトリップが正常に動作すること
        
        プロパティ11, 12
        """
        original_experiment = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": "breast_cancer",
            "model_type": "gradient_boosting",
            "accuracy": 0.97,
            "f1_score": 0.96,
            "hyperparameters": {
                "n_estimators": 150,
                "learning_rate": 0.1,
                "max_depth": 4
            },
            "training_time": 2.3
        }
        
        # 保存
        save_response = test_client.post("/api/experiments", json=original_experiment)
        assert save_response.status_code == 200
        experiment_id = save_response.json()["id"]
        
        # 取得
        get_response = test_client.get("/api/experiments")
        assert get_response.status_code == 200
        
        experiments = get_response.json()["experiments"]
        saved_experiment = next((e for e in experiments if e["id"] == experiment_id), None)
        
        assert saved_experiment is not None
        assert saved_experiment["dataset_name"] == original_experiment["dataset_name"]
        assert saved_experiment["model_type"] == original_experiment["model_type"]
        assert saved_experiment["accuracy"] == original_experiment["accuracy"]
        assert saved_experiment["f1_score"] == original_experiment["f1_score"]
        assert saved_experiment["hyperparameters"] == original_experiment["hyperparameters"]
        assert saved_experiment["training_time"] == original_experiment["training_time"]


class TestEndToEndWorkflow:
    """エンドツーエンドワークフローの統合テスト"""
    
    def test_complete_ml_workflow(self, test_client):
        """完全な機械学習ワークフローが正常に動作すること
        
        1. データセット一覧取得
        2. データセットプレビュー取得
        3. モデル学習
        4. モデル情報取得
        5. 実験記録保存
        6. 実験履歴取得
        7. モデルエクスポート
        
        要件: 9.1, 9.2, 9.3
        """
        # 1. データセット一覧取得
        datasets_response = test_client.get("/api/datasets")
        assert datasets_response.status_code == 200
        datasets = datasets_response.json()["datasets"]
        assert len(datasets) > 0
        
        # 2. データセットプレビュー取得
        preview_response = test_client.get("/api/datasets/iris/preview?n_rows=5")
        assert preview_response.status_code == 200
        
        # 3. モデル学習
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
        train_response = test_client.post("/api/train", json=training_config)
        assert train_response.status_code == 200
        train_data = train_response.json()
        model_id = train_data["model_id"]
        
        # 4. モデル情報取得
        model_info_response = test_client.get(f"/api/models/{model_id}")
        assert model_info_response.status_code == 200
        
        # 5. 実験記録保存
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": training_config["dataset_name"],
            "model_type": training_config["model_type"],
            "accuracy": train_data["accuracy"],
            "f1_score": train_data["f1_score"],
            "hyperparameters": training_config["hyperparameters"],
            "training_time": train_data["training_time"]
        }
        save_exp_response = test_client.post("/api/experiments", json=experiment)
        assert save_exp_response.status_code == 200
        
        # 6. 実験履歴取得
        get_exp_response = test_client.get("/api/experiments")
        assert get_exp_response.status_code == 200
        experiments = get_exp_response.json()["experiments"]
        assert len(experiments) > 0
        
        # 7. モデルエクスポート
        export_response = test_client.get(f"/api/models/{model_id}/export")
        assert export_response.status_code == 200
        assert len(export_response.content) > 0
    
    def test_multiple_models_comparison(self, test_client):
        """複数モデルの比較ワークフローが正常に動作すること
        
        要件: 5.3, 5.4
        """
        models_to_test = [
            ("random_forest", {"n_estimators": 10, "random_state": 42}),
            ("logistic_regression", {"C": 1.0, "random_state": 42}),
            ("gradient_boosting", {"n_estimators": 10, "learning_rate": 0.1, "random_state": 42})
        ]
        
        results = []
        
        for model_type, hyperparameters in models_to_test:
            # モデル学習
            training_config = {
                "dataset_name": "iris",
                "test_size": 0.3,
                "random_state": 42,
                "model_type": model_type,
                "hyperparameters": hyperparameters
            }
            
            train_response = test_client.post("/api/train", json=training_config)
            assert train_response.status_code == 200
            train_data = train_response.json()
            
            # 実験記録保存
            experiment = {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": "iris",
                "model_type": model_type,
                "accuracy": train_data["accuracy"],
                "f1_score": train_data["f1_score"],
                "hyperparameters": hyperparameters,
                "training_time": train_data["training_time"]
            }
            
            save_response = test_client.post("/api/experiments", json=experiment)
            assert save_response.status_code == 200
            
            results.append(train_data["accuracy"])
        
        # 実験履歴を取得
        get_response = test_client.get("/api/experiments")
        assert get_response.status_code == 200
        experiments = get_response.json()["experiments"]
        
        # 複数の実験が記録されていること
        assert len(experiments) >= len(models_to_test)
        
        # 最良モデルを識別できること（プロパティ14）
        best_accuracy = max(exp["accuracy"] for exp in experiments)
        best_experiment = next(exp for exp in experiments if exp["accuracy"] == best_accuracy)
        assert best_experiment is not None
        assert best_experiment["accuracy"] == best_accuracy
