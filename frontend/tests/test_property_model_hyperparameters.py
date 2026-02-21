"""
モデル別ハイパーパラメータ表示プロパティテスト

プロパティ5: モデル別ハイパーパラメータ表示
"""
import pytest
from hypothesis import given, strategies as st, settings


# **Feature: ml-visualization-dashboard, Property 5: モデル別ハイパーパラメータ表示**
@given(
    model_type=st.sampled_from([
        "random_forest",
        "gradient_boosting",
        "svm",
        "logistic_regression",
        "knn"
    ])
)
@settings(max_examples=100, deadline=None)
def test_property_model_specific_hyperparameters(model_type):
    """
    プロパティ5: モデル別ハイパーパラメータ表示
    
    任意のモデルタイプに対して、表示されるハイパーパラメータセットが
    そのモデルタイプに適切なパラメータのみを含むこと
    （例: Random Forestにはn_estimators、max_depthが含まれ、learning_rateは含まれない）
    
    検証: 要件 2.1
    """
    from src.components.sidebar import MODEL_HYPERPARAMETERS, render_hyperparameters
    from unittest.mock import MagicMock, patch
    
    # 期待されるハイパーパラメータセット（モデルタイプごと）
    expected_params = {
        "random_forest": {"n_estimators", "max_depth", "min_samples_split"},
        "gradient_boosting": {"n_estimators", "max_depth", "min_samples_split", "learning_rate"},
        "svm": {"C"},
        "logistic_regression": {"C"},
        "knn": {"k"}
    }
    
    # すべてのハイパーパラメータの集合
    all_possible_params = {
        "n_estimators", "max_depth", "min_samples_split", 
        "learning_rate", "C", "k"
    }
    
    # Streamlitのモックを作成
    mock_sidebar = MagicMock()
    
    # slider呼び出しを追跡するための辞書
    rendered_params = {}
    
    def mock_slider(display_name, min_value, max_value, value, step, key, **kwargs):
        # keyからパラメータ名を抽出（"hyperparam_" プレフィックスを削除）
        param_name = key.replace("hyperparam_", "")
        rendered_params[param_name] = value
        return value
    
    mock_sidebar.slider = mock_slider
    
    # streamlit.sidebarをモック
    with patch('src.components.sidebar.st.sidebar', mock_sidebar):
        # render_hyperparametersを呼び出し
        result = render_hyperparameters(model_type)
    
    # 返されたハイパーパラメータのキーセット
    returned_params = set(result.keys())
    
    # 期待されるパラメータセット
    expected_for_model = expected_params[model_type]
    
    # アサーション1: 返されたパラメータが期待されるパラメータと完全に一致すること
    assert returned_params == expected_for_model, (
        f"モデル '{model_type}' のハイパーパラメータが期待と一致しない: "
        f"期待={expected_for_model}, 実際={returned_params}, "
        f"不足={expected_for_model - returned_params}, "
        f"余分={returned_params - expected_for_model}"
    )
    
    # アサーション2: 他のモデルのパラメータが含まれていないこと
    other_model_params = all_possible_params - expected_for_model
    unexpected_params = returned_params & other_model_params
    
    assert len(unexpected_params) == 0, (
        f"モデル '{model_type}' に不適切なハイパーパラメータが含まれている: "
        f"{unexpected_params}"
    )
    
    # アサーション3: MODEL_HYPERPARAMETERSの定義と一致すること
    config_params = set(MODEL_HYPERPARAMETERS[model_type].keys())
    assert returned_params == config_params, (
        f"render_hyperparametersの返り値がMODEL_HYPERPARAMETERSの定義と一致しない: "
        f"期待={config_params}, 実際={returned_params}"
    )


@given(
    model_type=st.sampled_from([
        "random_forest",
        "gradient_boosting",
        "svm",
        "logistic_regression",
        "knn"
    ])
)
@settings(max_examples=100, deadline=None)
def test_property_hyperparameter_exclusivity(model_type):
    """
    プロパティ5の拡張: ハイパーパラメータの排他性
    
    任意のモデルタイプに対して、そのモデルに属さないハイパーパラメータが
    表示されないことを確認する
    
    検証: 要件 2.1
    """
    from src.components.sidebar import MODEL_HYPERPARAMETERS
    
    # 各モデルタイプに対する禁止パラメータ
    forbidden_params = {
        "random_forest": {"learning_rate", "C", "k"},
        "gradient_boosting": {"C", "k"},
        "svm": {"n_estimators", "max_depth", "min_samples_split", "learning_rate", "k"},
        "logistic_regression": {"n_estimators", "max_depth", "min_samples_split", "learning_rate", "k"},
        "knn": {"n_estimators", "max_depth", "min_samples_split", "learning_rate", "C"}
    }
    
    # モデルの設定を取得
    model_config = MODEL_HYPERPARAMETERS[model_type]
    actual_params = set(model_config.keys())
    
    # 禁止パラメータが含まれていないことを確認
    forbidden_for_model = forbidden_params[model_type]
    violations = actual_params & forbidden_for_model
    
    assert len(violations) == 0, (
        f"モデル '{model_type}' に不適切なハイパーパラメータが定義されている: "
        f"{violations}"
    )


def test_all_models_have_hyperparameters():
    """
    すべてのサポートされているモデルタイプがハイパーパラメータ定義を持つことを確認
    
    これは単体テストで、MODEL_HYPERPARAMETERSの完全性を検証する
    """
    from src.components.sidebar import MODEL_HYPERPARAMETERS
    
    supported_models = [
        "random_forest",
        "gradient_boosting",
        "svm",
        "logistic_regression",
        "knn"
    ]
    
    for model_type in supported_models:
        assert model_type in MODEL_HYPERPARAMETERS, (
            f"モデル '{model_type}' のハイパーパラメータ定義が存在しない"
        )
        
        assert len(MODEL_HYPERPARAMETERS[model_type]) > 0, (
            f"モデル '{model_type}' のハイパーパラメータが空"
        )


def test_hyperparameter_config_structure():
    """
    ハイパーパラメータ設定が正しい構造を持つことを確認
    
    各パラメータがmin, max, default, stepを持つことを検証する
    """
    from src.components.sidebar import MODEL_HYPERPARAMETERS
    
    required_keys = {"min", "max", "default", "step"}
    
    for model_type, params in MODEL_HYPERPARAMETERS.items():
        for param_name, config in params.items():
            actual_keys = set(config.keys())
            
            assert required_keys.issubset(actual_keys), (
                f"モデル '{model_type}' のパラメータ '{param_name}' に "
                f"必要なキーが不足している: "
                f"必要={required_keys}, 実際={actual_keys}, "
                f"不足={required_keys - actual_keys}"
            )
            
            # 値の型チェック
            assert isinstance(config["min"], (int, float)), (
                f"'{param_name}' の min が数値でない"
            )
            assert isinstance(config["max"], (int, float)), (
                f"'{param_name}' の max が数値でない"
            )
            assert isinstance(config["default"], (int, float)), (
                f"'{param_name}' の default が数値でない"
            )
            assert isinstance(config["step"], (int, float)), (
                f"'{param_name}' の step が数値でない"
            )
            
            # 範囲の妥当性チェック
            assert config["min"] < config["max"], (
                f"'{param_name}' の min が max 以上"
            )
            assert config["min"] <= config["default"] <= config["max"], (
                f"'{param_name}' の default が範囲外"
            )


def test_specific_model_hyperparameters():
    """
    特定のモデルが期待されるハイパーパラメータを持つことを確認
    
    要件2.2-2.7に基づく具体的な検証
    """
    from src.components.sidebar import MODEL_HYPERPARAMETERS
    
    # Random Forest (要件 2.2, 2.3, 2.4)
    rf_params = MODEL_HYPERPARAMETERS["random_forest"]
    assert "n_estimators" in rf_params, "Random Forest に n_estimators がない"
    assert "max_depth" in rf_params, "Random Forest に max_depth がない"
    assert "min_samples_split" in rf_params, "Random Forest に min_samples_split がない"
    assert "learning_rate" not in rf_params, "Random Forest に learning_rate が含まれている"
    
    # Gradient Boosting (要件 2.2, 2.3, 2.4, 2.5)
    gb_params = MODEL_HYPERPARAMETERS["gradient_boosting"]
    assert "n_estimators" in gb_params, "Gradient Boosting に n_estimators がない"
    assert "max_depth" in gb_params, "Gradient Boosting に max_depth がない"
    assert "min_samples_split" in gb_params, "Gradient Boosting に min_samples_split がない"
    assert "learning_rate" in gb_params, "Gradient Boosting に learning_rate がない"
    
    # SVM (要件 2.6)
    svm_params = MODEL_HYPERPARAMETERS["svm"]
    assert "C" in svm_params, "SVM に C がない"
    assert "n_estimators" not in svm_params, "SVM に n_estimators が含まれている"
    
    # Logistic Regression (要件 2.6)
    lr_params = MODEL_HYPERPARAMETERS["logistic_regression"]
    assert "C" in lr_params, "Logistic Regression に C がない"
    assert "n_estimators" not in lr_params, "Logistic Regression に n_estimators が含まれている"
    
    # KNN (要件 2.7)
    knn_params = MODEL_HYPERPARAMETERS["knn"]
    assert "k" in knn_params, "KNN に k がない"
    assert "C" not in knn_params, "KNN に C が含まれている"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
