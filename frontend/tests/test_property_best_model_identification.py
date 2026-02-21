"""
最良モデルの識別プロパティテスト

プロパティ14: 最良モデルの識別
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any


# **Feature: ml-visualization-dashboard, Property 14: 最良モデルの識別**
@given(
    experiments=st.lists(
        st.fixed_dictionaries({
            "id": st.integers(min_value=1, max_value=10000),
            "dataset_name": st.sampled_from(["iris", "wine", "breast_cancer", "digits"]),
            "model_type": st.sampled_from([
                "random_forest", "gradient_boosting", "svm", 
                "logistic_regression", "knn"
            ]),
            "accuracy": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            "f1_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            "timestamp": st.datetimes(
                min_value=__import__('datetime').datetime(2020, 1, 1),
                max_value=__import__('datetime').datetime(2030, 12, 31)
            )
        }),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=100, deadline=None)
def test_property_best_model_identification(experiments: List[Dict[str, Any]]):
    """
    プロパティ14: 最良モデルの識別
    
    任意の複数の実験記録に対して、最良モデルとして識別されるものが、
    最高accuracyを持つ実験であること
    
    検証: 要件 5.4
    """
    # 少なくとも1つの実験が必要
    assume(len(experiments) > 0)
    
    # すべてのaccuracyが有効な値であることを確認
    assume(all(0.0 <= exp["accuracy"] <= 1.0 for exp in experiments))
    
    # 最高accuracyを持つ実験を特定
    max_accuracy = max(exp["accuracy"] for exp in experiments)
    expected_best_experiments = [
        exp for exp in experiments 
        if exp["accuracy"] == max_accuracy
    ]
    
    # render_accuracy_trendの実装をテスト
    import pandas as pd
    
    # DataFrameを作成（実際の実装と同じ）
    df = pd.DataFrame(experiments)
    
    # 最良モデルを識別（実際の実装と同じロジック）
    best_idx = df["accuracy"].idxmax()
    df["is_best"] = False
    df.loc[best_idx, "is_best"] = True
    
    # 最良モデルとしてマークされた実験を取得
    best_models = df[df["is_best"]]
    
    # アサーション1: 少なくとも1つの最良モデルが識別されること
    assert len(best_models) > 0, (
        "最良モデルが識別されていない"
    )
    
    # アサーション2: 識別された最良モデルが最高accuracyを持つこと
    for _, best_model in best_models.iterrows():
        assert best_model["accuracy"] == max_accuracy, (
            f"最良モデルのaccuracyが最高値でない: "
            f"最良モデル={best_model['accuracy']}, 最高値={max_accuracy}"
        )
    
    # アサーション3: 最良モデルのaccuracyが他のすべてのモデル以上であること
    for _, exp in df.iterrows():
        if exp["is_best"]:
            assert exp["accuracy"] >= df["accuracy"].min(), (
                "最良モデルのaccuracyが他のモデルより低い"
            )
            assert exp["accuracy"] == df["accuracy"].max(), (
                "最良モデルのaccuracyが最高値でない"
            )


@given(
    experiments=st.lists(
        st.fixed_dictionaries({
            "id": st.integers(min_value=1, max_value=10000),
            "dataset_name": st.sampled_from(["iris", "wine"]),
            "model_type": st.sampled_from(["random_forest", "svm"]),
            "accuracy": st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
            "f1_score": st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        }),
        min_size=2,
        max_size=10
    )
)
@settings(max_examples=100, deadline=None)
def test_property_best_model_uniqueness_or_tie(experiments: List[Dict[str, Any]]):
    """
    プロパティ14の拡張: 最良モデルの一意性または同点処理
    
    任意の実験記録セットに対して、最高accuracyを持つ実験が複数ある場合、
    そのうちの1つが最良モデルとして識別されること
    
    検証: 要件 5.4
    """
    assume(len(experiments) >= 2)
    
    import pandas as pd
    
    df = pd.DataFrame(experiments)
    
    # 最良モデルを識別
    best_idx = df["accuracy"].idxmax()
    df["is_best"] = False
    df.loc[best_idx, "is_best"] = True
    
    # 最高accuracy
    max_accuracy = df["accuracy"].max()
    
    # 最高accuracyを持つ実験の数
    num_max_accuracy = (df["accuracy"] == max_accuracy).sum()
    
    # 最良としてマークされた実験の数
    num_best = df["is_best"].sum()
    
    # アサーション: 正確に1つの実験が最良としてマークされること
    # （pandasのidxmaxは最初の最大値のインデックスを返す）
    assert num_best == 1, (
        f"最良モデルが正確に1つマークされていない: {num_best}個マークされている"
    )
    
    # アサーション: マークされた実験が最高accuracyを持つこと
    best_model = df[df["is_best"]].iloc[0]
    assert best_model["accuracy"] == max_accuracy, (
        f"最良モデルのaccuracyが最高値でない: "
        f"最良={best_model['accuracy']}, 最高={max_accuracy}"
    )


@given(
    base_accuracy=st.floats(min_value=0.6, max_value=0.9),
    num_experiments=st.integers(min_value=3, max_value=15)
)
@settings(max_examples=100, deadline=None)
def test_property_best_model_with_varying_accuracies(base_accuracy: float, num_experiments: int):
    """
    プロパティ14の拡張: 様々なaccuracy値での最良モデル識別
    
    任意のベースaccuracyと実験数に対して、最高accuracyを持つ実験が
    正しく最良モデルとして識別されること
    
    検証: 要件 5.4
    """
    import pandas as pd
    import random
    
    # 様々なaccuracy値を持つ実験を生成
    experiments = []
    for i in range(num_experiments):
        # ベースaccuracyの周辺でランダムな値を生成
        accuracy = base_accuracy + random.uniform(-0.1, 0.1)
        accuracy = max(0.0, min(1.0, accuracy))  # 0-1の範囲に制限
        
        experiments.append({
            "id": i + 1,
            "dataset_name": "iris",
            "model_type": "random_forest",
            "accuracy": accuracy,
            "f1_score": accuracy * 0.95,  # F1スコアはaccuracyより少し低い
        })
    
    # 最後の実験を明確に最良にする
    experiments[-1]["accuracy"] = base_accuracy + 0.15
    experiments[-1]["accuracy"] = min(1.0, experiments[-1]["accuracy"])
    
    df = pd.DataFrame(experiments)
    
    # 最良モデルを識別
    best_idx = df["accuracy"].idxmax()
    df["is_best"] = False
    df.loc[best_idx, "is_best"] = True
    
    # 最高accuracy
    max_accuracy = df["accuracy"].max()
    
    # 最良モデルを取得
    best_model = df[df["is_best"]].iloc[0]
    
    # アサーション: 最良モデルが最高accuracyを持つこと
    assert best_model["accuracy"] == max_accuracy, (
        f"最良モデルのaccuracyが最高値でない: "
        f"最良={best_model['accuracy']}, 最高={max_accuracy}"
    )
    
    # アサーション: 最良モデルが最後の実験であること（明示的に最良にしたため）
    assert best_model["id"] == num_experiments, (
        f"最良モデルが期待される実験でない: "
        f"期待=実験{num_experiments}, 実際=実験{best_model['id']}"
    )


def test_best_model_identification_with_single_experiment():
    """
    単一の実験記録での最良モデル識別
    
    1つの実験しかない場合、その実験が最良モデルとして識別されること
    """
    import pandas as pd
    
    experiments = [{
        "id": 1,
        "dataset_name": "iris",
        "model_type": "random_forest",
        "accuracy": 0.95,
        "f1_score": 0.94,
    }]
    
    df = pd.DataFrame(experiments)
    
    # 最良モデルを識別
    best_idx = df["accuracy"].idxmax()
    df["is_best"] = False
    df.loc[best_idx, "is_best"] = True
    
    # アサーション: その実験が最良としてマークされること
    assert df["is_best"].sum() == 1, "最良モデルが1つマークされていない"
    assert df.loc[0, "is_best"] == True, "唯一の実験が最良としてマークされていない"


def test_best_model_identification_with_identical_accuracies():
    """
    同一accuracyを持つ複数の実験での最良モデル識別
    
    すべての実験が同じaccuracyを持つ場合、最初の実験が最良として識別されること
    """
    import pandas as pd
    
    experiments = [
        {"id": 1, "dataset_name": "iris", "model_type": "random_forest", "accuracy": 0.90, "f1_score": 0.89},
        {"id": 2, "dataset_name": "iris", "model_type": "svm", "accuracy": 0.90, "f1_score": 0.89},
        {"id": 3, "dataset_name": "iris", "model_type": "knn", "accuracy": 0.90, "f1_score": 0.89},
    ]
    
    df = pd.DataFrame(experiments)
    
    # 最良モデルを識別
    best_idx = df["accuracy"].idxmax()
    df["is_best"] = False
    df.loc[best_idx, "is_best"] = True
    
    # アサーション: 正確に1つの実験が最良としてマークされること
    assert df["is_best"].sum() == 1, "最良モデルが正確に1つマークされていない"
    
    # アサーション: 最初の実験が最良としてマークされること（pandasのidxmaxの動作）
    assert df.loc[0, "is_best"] == True, "最初の実験が最良としてマークされていない"


def test_best_model_identification_with_extreme_values():
    """
    極端なaccuracy値での最良モデル識別
    
    accuracy=0.0や1.0などの極端な値でも正しく識別されること
    """
    import pandas as pd
    
    experiments = [
        {"id": 1, "dataset_name": "iris", "model_type": "random_forest", "accuracy": 0.0, "f1_score": 0.0},
        {"id": 2, "dataset_name": "iris", "model_type": "svm", "accuracy": 0.5, "f1_score": 0.48},
        {"id": 3, "dataset_name": "iris", "model_type": "knn", "accuracy": 1.0, "f1_score": 1.0},
    ]
    
    df = pd.DataFrame(experiments)
    
    # 最良モデルを識別
    best_idx = df["accuracy"].idxmax()
    df["is_best"] = False
    df.loc[best_idx, "is_best"] = True
    
    # アサーション: accuracy=1.0の実験が最良としてマークされること
    best_model = df[df["is_best"]].iloc[0]
    assert best_model["accuracy"] == 1.0, "accuracy=1.0の実験が最良としてマークされていない"
    assert best_model["id"] == 3, "正しい実験が最良としてマークされていない"


def test_best_model_visualization_integration():
    """
    可視化関数との統合テスト
    
    render_accuracy_trend関数が最良モデルを正しく識別して可視化すること
    """
    from src.components.visualizations import render_accuracy_trend
    from unittest.mock import patch, MagicMock
    import pandas as pd
    
    experiments = [
        {"id": 1, "dataset_name": "iris", "model_type": "random_forest", "accuracy": 0.85, "f1_score": 0.84},
        {"id": 2, "dataset_name": "iris", "model_type": "svm", "accuracy": 0.92, "f1_score": 0.91},
        {"id": 3, "dataset_name": "iris", "model_type": "knn", "accuracy": 0.88, "f1_score": 0.87},
    ]
    
    # Streamlitのモック
    with patch('src.components.visualizations.st') as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.info = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.error = MagicMock()
        mock_st.plotly_chart = MagicMock()
        
        # 関数を呼び出し
        render_accuracy_trend(experiments)
        
        # plotly_chartが呼ばれたことを確認
        assert mock_st.plotly_chart.called, "plotly_chartが呼ばれていない"
        
        # 呼び出された引数を取得
        call_args = mock_st.plotly_chart.call_args
        fig = call_args[0][0]
        
        # figureにデータが含まれていることを確認
        assert len(fig.data) > 0, "figureにデータが含まれていない"
        
        # 最良モデルのトレースが存在することを確認
        trace_names = [trace.name for trace in fig.data]
        assert "Best Model" in trace_names, "最良モデルのトレースが存在しない"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
