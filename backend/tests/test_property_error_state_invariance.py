"""
プロパティベーステスト: エラー時の状態不変性

**Feature: ml-visualization-dashboard, Property 9: エラー時の状態不変性**
**検証: 要件 3.4**
"""
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st, settings

from backend.src.main import app, trained_models


# テストクライアント
client = TestClient(app, raise_server_exceptions=False)


class TestPropertyErrorStateInvariance:
    """プロパティベーステスト: エラー時の状態不変性"""
    
    # **Feature: ml-visualization-dashboard, Property 9: エラー時の状態不変性**
    @given(
        invalid_dataset_name=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in ["iris", "wine", "breast_cancer", "digits"]
        ),
        model_type=st.sampled_from(["random_forest", "gradient_boosting", "svm", "logistic_regression", "knn"])
    )
    @settings(max_examples=100, deadline=None)
    def test_error_preserves_previous_state_invalid_dataset(self, invalid_dataset_name, model_type):
        """
        プロパティ9: 任意の無効な学習設定（存在しないデータセット名）に対して、
        学習を試みた場合、エラーが返され、以前の学習結果が変更されないこと
        検証: 要件 3.4
        """
        # 初期状態を記録（学習済みモデルの数）
        initial_model_count = len(trained_models)
        initial_model_ids = set(trained_models.keys())
        
        # 有効な学習を実行して初期状態を作成（初回のみ）
        if initial_model_count == 0:
            valid_config = {
                "dataset_name": "iris",
                "test_size": 0.3,
                "random_state": 42,
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            }
            response = client.post("/api/train", json=valid_config)
            assert response.status_code == 200, "Initial valid training should succeed"
            
            # 初期状態を再記録
            initial_model_count = len(trained_models)
            initial_model_ids = set(trained_models.keys())
            
            # 初期状態の学習結果を保存
            initial_result = response.json()
        else:
            # 既存のモデルから1つ選んで初期結果とする
            initial_model_id = list(trained_models.keys())[0]
            initial_result = {
                "model_id": initial_model_id,
                "dataset_name": trained_models[initial_model_id]["dataset_name"],
                "model_type": trained_models[initial_model_id]["model_type"]
            }
        
        # 無効なデータセット名で学習を試みる
        invalid_config = {
            "dataset_name": invalid_dataset_name,
            "test_size": 0.3,
            "random_state": 42,
            "model_type": model_type,
            "hyperparameters": {"n_estimators": 10, "random_state": 42}
        }
        
        response = client.post("/api/train", json=invalid_config)
        
        # プロパティ検証1: エラーが返されること
        assert response.status_code in [404, 500], \
            f"Expected error status code (404 or 500), got {response.status_code}"
        
        # プロパティ検証2: 学習済みモデルの数が変わらないこと
        assert len(trained_models) == initial_model_count, \
            f"Model count changed after error. Before: {initial_model_count}, After: {len(trained_models)}"
        
        # プロパティ検証3: 既存のモデルIDが変更されていないこと
        current_model_ids = set(trained_models.keys())
        assert current_model_ids == initial_model_ids, \
            f"Model IDs changed after error. Before: {initial_model_ids}, After: {current_model_ids}"
        
        # プロパティ検証4: 初期状態のモデルデータが変更されていないこと
        if initial_model_count > 0:
            # 最初のモデルのデータが保持されていることを確認
            first_model_id = list(initial_model_ids)[0]
            assert first_model_id in trained_models, \
                f"Initial model {first_model_id} was removed after error"
            
            # モデルデータの整合性チェック
            model_data = trained_models[first_model_id]
            assert "model" in model_data, "Model object missing from stored data"
            assert "dataset_name" in model_data, "Dataset name missing from stored data"
            assert "model_type" in model_data, "Model type missing from stored data"
            assert "config" in model_data, "Config missing from stored data"
    
    # **Feature: ml-visualization-dashboard, Property 9: エラー時の状態不変性**
    @given(
        test_size=st.floats(min_value=-1.0, max_value=-0.01).filter(lambda x: x < 0) | 
                  st.floats(min_value=0.51, max_value=2.0).filter(lambda x: x > 0.5)
    )
    @settings(max_examples=100, deadline=None)
    def test_error_preserves_previous_state_invalid_test_size(self, test_size):
        """
        プロパティ9: 任意の無効なtest_size（範囲外の値）に対して、
        学習を試みた場合、エラーが返され、以前の学習結果が変更されないこと
        検証: 要件 3.4
        """
        # 初期状態を記録
        initial_model_count = len(trained_models)
        initial_model_ids = set(trained_models.keys())
        
        # 有効な学習を実行して初期状態を作成（初回のみ）
        if initial_model_count == 0:
            valid_config = {
                "dataset_name": "iris",
                "test_size": 0.3,
                "random_state": 42,
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            }
            response = client.post("/api/train", json=valid_config)
            assert response.status_code == 200, "Initial valid training should succeed"
            
            # 初期状態を再記録
            initial_model_count = len(trained_models)
            initial_model_ids = set(trained_models.keys())
        
        # 無効なtest_sizeで学習を試みる
        invalid_config = {
            "dataset_name": "iris",
            "test_size": test_size,
            "random_state": 42,
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10, "random_state": 42}
        }
        
        response = client.post("/api/train", json=invalid_config)
        
        # プロパティ検証1: エラーが返されること（バリデーションエラー）
        assert response.status_code == 422, \
            f"Expected validation error status code (422), got {response.status_code}"
        
        # プロパティ検証2: 学習済みモデルの数が変わらないこと
        assert len(trained_models) == initial_model_count, \
            f"Model count changed after error. Before: {initial_model_count}, After: {len(trained_models)}"
        
        # プロパティ検証3: 既存のモデルIDが変更されていないこと
        current_model_ids = set(trained_models.keys())
        assert current_model_ids == initial_model_ids, \
            f"Model IDs changed after error. Before: {initial_model_ids}, After: {current_model_ids}"
    
    # **Feature: ml-visualization-dashboard, Property 9: エラー時の状態不変性**
    @given(
        random_state=st.integers(min_value=-1000, max_value=-1)
    )
    @settings(max_examples=100, deadline=None)
    def test_error_preserves_previous_state_invalid_random_state(self, random_state):
        """
        プロパティ9: 任意の無効なrandom_state（負の値）に対して、
        学習を試みた場合、エラーが返され、以前の学習結果が変更されないこと
        検証: 要件 3.4
        """
        # 初期状態を記録
        initial_model_count = len(trained_models)
        initial_model_ids = set(trained_models.keys())
        
        # 有効な学習を実行して初期状態を作成（初回のみ）
        if initial_model_count == 0:
            valid_config = {
                "dataset_name": "iris",
                "test_size": 0.3,
                "random_state": 42,
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            }
            response = client.post("/api/train", json=valid_config)
            assert response.status_code == 200, "Initial valid training should succeed"
            
            # 初期状態を再記録
            initial_model_count = len(trained_models)
            initial_model_ids = set(trained_models.keys())
        
        # 無効なrandom_stateで学習を試みる
        invalid_config = {
            "dataset_name": "iris",
            "test_size": 0.3,
            "random_state": random_state,
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10, "random_state": 42}
        }
        
        response = client.post("/api/train", json=invalid_config)
        
        # プロパティ検証1: エラーが返されること（バリデーションエラー）
        assert response.status_code == 422, \
            f"Expected validation error status code (422), got {response.status_code}"
        
        # プロパティ検証2: 学習済みモデルの数が変わらないこと
        assert len(trained_models) == initial_model_count, \
            f"Model count changed after error. Before: {initial_model_count}, After: {len(trained_models)}"
        
        # プロパティ検証3: 既存のモデルIDが変更されていないこと
        current_model_ids = set(trained_models.keys())
        assert current_model_ids == initial_model_ids, \
            f"Model IDs changed after error. Before: {initial_model_ids}, After: {current_model_ids}"
    
    def test_cleanup_models_after_tests(self):
        """テスト後のクリーンアップ"""
        # すべてのモデルをクリア
        trained_models.clear()
        assert len(trained_models) == 0, "Models should be cleared after tests"
