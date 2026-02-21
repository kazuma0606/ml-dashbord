"""
プロパティベーステスト: APIレスポンスの適切性

**Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
**検証: 要件 8.6**
"""
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st, settings
from unittest.mock import patch, MagicMock

from src.main import app
from src.services.cache import CacheManager


# モックRedisクライアントを作成
class MockRedis:
    """テスト用モックRedisクライアント"""
    def __init__(self):
        self.data = {}
    
    def ping(self):
        return True
    
    def get(self, key):
        return self.data.get(key)
    
    def setex(self, key, ttl, value):
        self.data[key] = value
        return True
    
    def exists(self, key):
        return 1 if key in self.data else 0
    
    def delete(self, key):
        if key in self.data:
            del self.data[key]
        return True
    
    def close(self):
        pass


# モックキャッシュマネージャーを作成
@pytest.fixture(autouse=True)
def mock_cache():
    """テスト用にRedis接続をモック"""
    mock_redis = MockRedis()
    
    with patch('src.services.cache.redis.Redis', return_value=mock_redis):
        # キャッシュマネージャーをリセット
        import src.services.cache as cache_module
        cache_module._cache_manager = None
        yield
        # テスト後にクリーンアップ
        cache_module._cache_manager = None


# テストクライアント
client = TestClient(app, raise_server_exceptions=False)


class TestPropertyAPIResponseAppropriateness:
    """プロパティベーステスト: APIレスポンスの適切性"""
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"]),
        test_size=st.floats(min_value=0.1, max_value=0.5),
        random_state=st.integers(min_value=0, max_value=10000),
        model_type=st.sampled_from(["random_forest", "gradient_boosting", "svm", "logistic_regression", "knn"]),
        n_estimators=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_train_request_returns_2xx_with_data(self, dataset_name, test_size, 
                                                        random_state, model_type, n_estimators):
        """
        プロパティ19: 任意の有効なAPIリクエスト（学習リクエスト）に対して、
        レスポンスが適切なステータスコード（2xx）とデータ構造を持つこと
        検証: 要件 8.6
        """
        # 有効な学習設定を構築
        config = {
            "dataset_name": dataset_name,
            "test_size": test_size,
            "random_state": random_state,
            "model_type": model_type,
            "hyperparameters": {}
        }
        
        # モデルタイプに応じたハイパーパラメータを設定
        if model_type in ["random_forest", "gradient_boosting"]:
            config["hyperparameters"]["n_estimators"] = n_estimators
            config["hyperparameters"]["random_state"] = random_state
        elif model_type == "svm":
            config["hyperparameters"]["random_state"] = random_state
        elif model_type == "logistic_regression":
            config["hyperparameters"]["random_state"] = random_state
        elif model_type == "knn":
            config["hyperparameters"]["n_neighbors"] = min(5, n_estimators)
        
        # APIリクエストを送信
        response = client.post("/api/train", json=config)
        
        # プロパティ検証1: ステータスコードが2xx（成功）であること
        assert 200 <= response.status_code < 300, \
            f"Expected 2xx status code for valid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスデータが適切な構造を持つこと
        data = response.json()
        
        # 必須フィールドの存在確認
        required_fields = ["model_id", "accuracy", "f1_score", "confusion_matrix", 
                          "classification_report", "training_time"]
        for field in required_fields:
            assert field in data, \
                f"Required field '{field}' missing from response data"
        
        # データ型の検証
        assert isinstance(data["model_id"], str), "model_id should be string"
        assert isinstance(data["accuracy"], (int, float)), "accuracy should be numeric"
        assert isinstance(data["f1_score"], (int, float)), "f1_score should be numeric"
        assert isinstance(data["confusion_matrix"], list), "confusion_matrix should be list"
        assert isinstance(data["classification_report"], dict), "classification_report should be dict"
        assert isinstance(data["training_time"], (int, float)), "training_time should be numeric"
        
        # 値の範囲検証
        assert 0.0 <= data["accuracy"] <= 1.0, \
            f"accuracy should be between 0 and 1, got {data['accuracy']}"
        assert 0.0 <= data["f1_score"] <= 1.0, \
            f"f1_score should be between 0 and 1, got {data['f1_score']}"
        assert data["training_time"] > 0, \
            f"training_time should be positive, got {data['training_time']}"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"]),
        n_rows=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_dataset_preview_request_returns_2xx_with_data(self, dataset_name, n_rows):
        """
        プロパティ19: 任意の有効なAPIリクエスト（データセットプレビュー）に対して、
        レスポンスが適切なステータスコード（2xx）とデータ構造を持つこと
        検証: 要件 8.6
        """
        # APIリクエストを送信
        response = client.get(f"/api/datasets/{dataset_name}/preview?n_rows={n_rows}")
        
        # プロパティ検証1: ステータスコードが2xx（成功）であること
        assert 200 <= response.status_code < 300, \
            f"Expected 2xx status code for valid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスデータが適切な構造を持つこと
        data = response.json()
        
        # 必須フィールドの存在確認
        required_fields = ["data", "columns", "n_rows"]
        for field in required_fields:
            assert field in data, \
                f"Required field '{field}' missing from response data"
        
        # データ型の検証
        assert isinstance(data["data"], list), "data should be list"
        assert isinstance(data["columns"], list), "columns should be list"
        assert isinstance(data["n_rows"], int), "n_rows should be integer"
        
        # 値の検証
        assert data["n_rows"] >= 0, f"n_rows should be non-negative, got {data['n_rows']}"
        assert len(data["columns"]) > 0, "columns should not be empty"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"])
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_dataset_metadata_request_returns_2xx_with_data(self, dataset_name):
        """
        プロパティ19: 任意の有効なAPIリクエスト（データセットメタデータ）に対して、
        レスポンスが適切なステータスコード（2xx）とデータ構造を持つこと
        検証: 要件 8.6
        """
        # APIリクエストを送信
        response = client.get(f"/api/datasets/{dataset_name}")
        
        # プロパティ検証1: ステータスコードが2xx（成功）であること
        assert 200 <= response.status_code < 300, \
            f"Expected 2xx status code for valid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスデータが適切な構造を持つこと
        data = response.json()
        
        # 必須フィールドの存在確認
        required_fields = ["name", "n_samples", "n_features"]
        for field in required_fields:
            assert field in data, \
                f"Required field '{field}' missing from response data"
        
        # データ型の検証
        assert isinstance(data["name"], str), "name should be string"
        assert isinstance(data["n_samples"], int), "n_samples should be integer"
        assert isinstance(data["n_features"], int), "n_features should be integer"
        
        # 値の検証
        assert data["name"] == dataset_name, \
            f"name should match requested dataset, expected {dataset_name}, got {data['name']}"
        assert data["n_samples"] > 0, f"n_samples should be positive, got {data['n_samples']}"
        assert data["n_features"] > 0, f"n_features should be positive, got {data['n_features']}"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    def test_valid_datasets_list_request_returns_2xx_with_data(self):
        """
        プロパティ19: 有効なAPIリクエスト（データセット一覧）に対して、
        レスポンスが適切なステータスコード（2xx）とデータ構造を持つこと
        検証: 要件 8.6
        """
        # APIリクエストを送信
        response = client.get("/api/datasets")
        
        # プロパティ検証1: ステータスコードが2xx（成功）であること
        assert 200 <= response.status_code < 300, \
            f"Expected 2xx status code for valid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスデータが適切な構造を持つこと
        data = response.json()
        
        # 必須フィールドの存在確認
        assert "datasets" in data, "Required field 'datasets' missing from response data"
        
        # データ型の検証
        assert isinstance(data["datasets"], list), "datasets should be list"
        assert len(data["datasets"]) > 0, "datasets should not be empty"
        
        # 各データセットの構造検証
        for dataset in data["datasets"]:
            assert isinstance(dataset, dict), "Each dataset should be a dict"
            assert "name" in dataset, "Each dataset should have 'name' field"
            assert "description" in dataset, "Each dataset should have 'description' field"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    @given(
        invalid_dataset_name=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in ["iris", "wine", "breast_cancer", "digits"] and x.isalnum()
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_dataset_request_returns_4xx_with_error(self, invalid_dataset_name):
        """
        プロパティ19: 任意の無効なAPIリクエスト（存在しないデータセット）に対して、
        エラーステータスコード（4xx）とエラーメッセージを持つこと
        検証: 要件 8.6
        """
        # APIリクエストを送信
        response = client.get(f"/api/datasets/{invalid_dataset_name}")
        
        # プロパティ検証1: ステータスコードが4xx（クライアントエラー）であること
        assert 400 <= response.status_code < 500, \
            f"Expected 4xx status code for invalid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスにエラーメッセージが含まれること
        data = response.json()
        assert "detail" in data, "Error response should contain 'detail' field"
        assert isinstance(data["detail"], str), "Error detail should be string"
        assert len(data["detail"]) > 0, "Error detail should not be empty"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    @given(
        test_size=st.floats(min_value=-1.0, max_value=-0.01) | 
                  st.floats(min_value=0.51, max_value=2.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_train_request_returns_4xx_with_error(self, test_size):
        """
        プロパティ19: 任意の無効なAPIリクエスト（無効なパラメータ）に対して、
        エラーステータスコード（4xx）とエラーメッセージを持つこと
        検証: 要件 8.6
        """
        # 無効な学習設定を構築
        config = {
            "dataset_name": "iris",
            "test_size": test_size,
            "random_state": 42,
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10, "random_state": 42}
        }
        
        # APIリクエストを送信
        response = client.post("/api/train", json=config)
        
        # プロパティ検証1: ステータスコードが4xx（クライアントエラー）であること
        assert 400 <= response.status_code < 500, \
            f"Expected 4xx status code for invalid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスにエラーメッセージが含まれること
        data = response.json()
        assert "detail" in data, "Error response should contain 'detail' field"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    @given(
        invalid_model_id=st.text(min_size=1, max_size=50).filter(lambda x: x.isalnum())
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_model_id_request_returns_4xx_with_error(self, invalid_model_id):
        """
        プロパティ19: 任意の無効なAPIリクエスト（存在しないモデルID）に対して、
        エラーステータスコード（4xx）とエラーメッセージを持つこと
        検証: 要件 8.6
        """
        # APIリクエストを送信
        response = client.get(f"/api/models/{invalid_model_id}")
        
        # プロパティ検証1: ステータスコードが4xx（クライアントエラー）であること
        assert 400 <= response.status_code < 500, \
            f"Expected 4xx status code for invalid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスにエラーメッセージが含まれること
        data = response.json()
        assert "detail" in data, "Error response should contain 'detail' field"
        assert isinstance(data["detail"], str), "Error detail should be string"
        assert len(data["detail"]) > 0, "Error detail should not be empty"
    
    # **Feature: ml-visualization-dashboard, Property 19: APIレスポンスの適切性**
    def test_health_check_returns_2xx_with_data(self):
        """
        プロパティ19: ヘルスチェックエンドポイントに対して、
        レスポンスが適切なステータスコード（2xx）とデータ構造を持つこと
        検証: 要件 8.6
        """
        # APIリクエストを送信
        response = client.get("/health")
        
        # プロパティ検証1: ステータスコードが2xx（成功）であること
        assert 200 <= response.status_code < 300, \
            f"Expected 2xx status code for valid request, got {response.status_code}"
        
        # プロパティ検証2: レスポンスがJSON形式であること
        assert response.headers.get("content-type") == "application/json", \
            f"Expected JSON response, got {response.headers.get('content-type')}"
        
        # プロパティ検証3: レスポンスデータが適切な構造を持つこと
        data = response.json()
        assert "status" in data, "Health check response should contain 'status' field"
        assert data["status"] == "healthy", "Health check status should be 'healthy'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
