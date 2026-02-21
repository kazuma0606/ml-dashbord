"""
機械学習サービスのテスト
"""
import pytest
import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.services.model_factory import ModelFactory
from src.services.model_trainer import ModelTrainer
from src.services.metrics_calculator import MetricsCalculator
from src.models.exceptions import ModelTrainingError

from hypothesis import given, strategies as st, settings


class TestModelFactory:
    """ModelFactoryクラスのテスト"""
    
    def test_create_random_forest(self):
        """Random Forestモデルの作成"""
        model = ModelFactory.create_model("random_forest", {"n_estimators": 50, "max_depth": 5})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    def test_create_gradient_boosting(self):
        """Gradient Boostingモデルの作成"""
        model = ModelFactory.create_model("gradient_boosting", {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        })
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
    
    def test_create_svm(self):
        """SVMモデルの作成"""
        model = ModelFactory.create_model("svm", {"C": 1.0, "kernel": "rbf"})
        assert isinstance(model, SVC)
        assert model.C == 1.0
        assert model.kernel == "rbf"
    
    def test_create_logistic_regression(self):
        """Logistic Regressionモデルの作成"""
        model = ModelFactory.create_model("logistic_regression", {"C": 0.5, "max_iter": 200})
        assert isinstance(model, LogisticRegression)
        assert model.C == 0.5
        assert model.max_iter == 200
    
    def test_create_knn(self):
        """KNNモデルの作成"""
        model = ModelFactory.create_model("knn", {"n_neighbors": 7})
        assert isinstance(model, KNeighborsClassifier)
        assert model.n_neighbors == 7
    
    def test_unsupported_model_type(self):
        """サポートされていないモデルタイプでエラー"""
        with pytest.raises(ModelTrainingError):
            ModelFactory.create_model("unsupported_model", {})
    
    def test_filter_hyperparameters(self):
        """ハイパーパラメータのフィルタリング"""
        # Random Forestに不適切なパラメータを含む
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,  # Random Forestには不要
            "C": 1.0  # Random Forestには不要
        }
        model = ModelFactory.create_model("random_forest", params)
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
        assert model.max_depth == 5


class TestModelTrainer:
    """ModelTrainerクラスのテスト"""
    
    def test_train_model(self):
        """モデル学習の基本テスト"""
        # テストデータ生成
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, y_train = X[:80], y[:80]
        
        # モデル作成と学習
        model = ModelFactory.create_model("random_forest", {"n_estimators": 10, "random_state": 42})
        trained_model, training_time = ModelTrainer.train(model, X_train, y_train)
        
        assert trained_model is not None
        assert training_time > 0
        assert hasattr(trained_model, 'predict')
    
    def test_training_time_measurement(self):
        """学習時間が正しく計測される"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, y_train = X[:80], y[:80]
        
        model = ModelFactory.create_model("logistic_regression", {"random_state": 42})
        trained_model, training_time = ModelTrainer.train(model, X_train, y_train)
        
        assert training_time >= 0
        assert isinstance(training_time, float)


class TestMetricsCalculator:
    """MetricsCalculatorクラスのテスト"""
    
    def test_calculate_accuracy(self):
        """Accuracy計算"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy = MetricsCalculator.calculate_accuracy(y_true, y_pred)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 0.8  # 5つ中4つ正解
    
    def test_calculate_f1_score(self):
        """F1スコア計算"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        f1 = MetricsCalculator.calculate_f1_score(y_true, y_pred)
        assert 0.0 <= f1 <= 1.0
    
    def test_generate_confusion_matrix(self):
        """混同行列生成"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        cm = MetricsCalculator.generate_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, list)
        assert len(cm) == 2  # 2クラス
        assert len(cm[0]) == 2
    
    def test_generate_classification_report(self):
        """分類レポート生成"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        report = MetricsCalculator.generate_classification_report(y_true, y_pred)
        assert isinstance(report, dict)
        assert 'accuracy' in report
        assert 'weighted avg' in report
    
    def test_extract_feature_importances_tree_model(self):
        """特徴量重要度抽出（木ベースモデル）"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importances = MetricsCalculator.extract_feature_importances(model)
        assert importances is not None
        assert len(importances) == 4
        # 降順にソートされているか確認
        assert importances == sorted(importances, reverse=True)
    
    def test_extract_feature_importances_non_tree_model(self):
        """特徴量重要度抽出（非木ベースモデル）"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        importances = MetricsCalculator.extract_feature_importances(model)
        assert importances is None



class TestPropertyModelTrainingSuccess:
    """プロパティベーステスト: モデル学習の成功"""
    
    # **Feature: ml-visualization-dashboard, Property 7: モデル学習の成功**
    @given(
        model_type=st.sampled_from(["random_forest", "gradient_boosting", "svm", "logistic_regression", "knn"]),
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=4, max_value=20),
        n_classes=st.integers(min_value=2, max_value=4),
        test_size=st.floats(min_value=0.1, max_value=0.5)
    )
    @settings(max_examples=100, deadline=None)
    def test_model_training_returns_metrics(self, model_type, n_samples, n_features, n_classes, test_size):
        """
        プロパティ7: 任意の有効なモデル設定とデータセットで学習が成功し評価指標が返される
        検証: 要件 3.1
        """
        # n_informativeを計算（n_classes * n_clusters_per_class <= 2^n_informative を満たす）
        # n_clusters_per_class=2 を使用するため、n_informative >= log2(n_classes * 2)
        import math
        min_n_informative = max(2, math.ceil(math.log2(n_classes * 2)))
        n_informative = min(n_features, max(min_n_informative, n_features // 2))
        
        # テストデータ生成
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            random_state=42
        )
        
        # データ分割
        split_idx = int(n_samples * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # モデル作成（シンプルなハイパーパラメータ）
        hyperparameters = {}
        if model_type in ["random_forest", "gradient_boosting"]:
            hyperparameters["n_estimators"] = 10  # 高速化のため少なめ
            hyperparameters["random_state"] = 42
        elif model_type in ["svm", "logistic_regression"]:
            hyperparameters["random_state"] = 42
            if model_type == "logistic_regression":
                hyperparameters["max_iter"] = 1000
        elif model_type == "knn":
            hyperparameters["n_neighbors"] = min(5, len(X_train) - 1)
        
        model = ModelFactory.create_model(model_type, hyperparameters)
        
        # モデル学習
        trained_model, training_time = ModelTrainer.train(model, X_train, y_train)
        
        # 予測
        y_pred = trained_model.predict(X_test)
        
        # 評価指標計算
        accuracy = MetricsCalculator.calculate_accuracy(y_test, y_pred)
        f1 = MetricsCalculator.calculate_f1_score(y_test, y_pred)
        
        # プロパティ検証: 評価指標が返されること
        assert accuracy is not None, "Accuracy should not be None"
        assert f1 is not None, "F1 score should not be None"
        assert isinstance(accuracy, float), "Accuracy should be a float"
        assert isinstance(f1, float), "F1 score should be a float"
        assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"
        assert 0.0 <= f1 <= 1.0, f"F1 score should be between 0 and 1, got {f1}"
        assert training_time > 0, "Training time should be positive"


class TestPropertyMetricsUpdate:
    """プロパティベーステスト: 学習完了後のメトリクス更新"""
    
    # **Feature: ml-visualization-dashboard, Property 8: 学習完了後のメトリクス更新**
    @given(
        model_type=st.sampled_from(["random_forest", "gradient_boosting", "svm", "logistic_regression", "knn"]),
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=4, max_value=20),
        n_classes=st.integers(min_value=2, max_value=4),
        test_size=st.floats(min_value=0.1, max_value=0.5)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_metrics_updated_after_training(self, model_type, n_samples, n_features, n_classes, test_size):
        """
        プロパティ8: 任意のモデル学習実行後、すべての必須メトリクスが更新されNone以外の値を持つ
        検証: 要件 3.3, 4.1, 4.2, 4.3, 4.4, 4.6
        """
        # n_informativeを計算（n_classes * n_clusters_per_class <= 2^n_informative を満たす）
        import math
        min_n_informative = max(2, math.ceil(math.log2(n_classes * 2)))
        n_informative = min(n_features, max(min_n_informative, n_features // 2))
        
        # テストデータ生成
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            random_state=42
        )
        
        # データ分割
        split_idx = int(n_samples * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # モデル作成
        hyperparameters = {}
        if model_type in ["random_forest", "gradient_boosting"]:
            hyperparameters["n_estimators"] = 10
            hyperparameters["random_state"] = 42
        elif model_type in ["svm", "logistic_regression"]:
            hyperparameters["random_state"] = 42
            if model_type == "logistic_regression":
                hyperparameters["max_iter"] = 1000
        elif model_type == "knn":
            hyperparameters["n_neighbors"] = min(5, len(X_train) - 1)
        
        model = ModelFactory.create_model(model_type, hyperparameters)
        
        # モデル学習
        trained_model, training_time = ModelTrainer.train(model, X_train, y_train)
        
        # 予測
        y_pred = trained_model.predict(X_test)
        
        # すべてのメトリクスを計算
        accuracy = MetricsCalculator.calculate_accuracy(y_test, y_pred)
        f1_score_value = MetricsCalculator.calculate_f1_score(y_test, y_pred)
        confusion_mat = MetricsCalculator.generate_confusion_matrix(y_test, y_pred)
        classification_rep = MetricsCalculator.generate_classification_report(y_test, y_pred)
        feature_importances = MetricsCalculator.extract_feature_importances(trained_model)
        
        # モデル名を取得
        model_name = type(trained_model).__name__
        
        # プロパティ検証: すべての必須メトリクスがNone以外の値を持つ
        assert accuracy is not None, "Accuracy should not be None"
        assert f1_score_value is not None, "F1 score should not be None"
        assert model_name is not None, "Model name should not be None"
        assert confusion_mat is not None, "Confusion matrix should not be None"
        assert classification_rep is not None, "Classification report should not be None"
        
        # 型チェック
        assert isinstance(accuracy, float), f"Accuracy should be float, got {type(accuracy)}"
        assert isinstance(f1_score_value, float), f"F1 score should be float, got {type(f1_score_value)}"
        assert isinstance(model_name, str), f"Model name should be string, got {type(model_name)}"
        assert isinstance(confusion_mat, list), f"Confusion matrix should be list, got {type(confusion_mat)}"
        assert isinstance(classification_rep, dict), f"Classification report should be dict, got {type(classification_rep)}"
        
        # 値の範囲チェック
        assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"
        assert 0.0 <= f1_score_value <= 1.0, f"F1 score should be between 0 and 1, got {f1_score_value}"
        
        # 混同行列の構造チェック
        assert len(confusion_mat) > 0, "Confusion matrix should not be empty"
        assert all(isinstance(row, list) for row in confusion_mat), "Confusion matrix rows should be lists"
        
        # 分類レポートの必須キーチェック
        assert 'accuracy' in classification_rep, "Classification report should contain 'accuracy'"
        assert 'weighted avg' in classification_rep, "Classification report should contain 'weighted avg'"
        
        # 特徴量重要度は木ベースモデルのみ（オプショナル）
        if model_type in ["random_forest", "gradient_boosting"]:
            assert feature_importances is not None, f"Feature importances should not be None for {model_type}"
            assert isinstance(feature_importances, list), "Feature importances should be a list"
            assert len(feature_importances) > 0, "Feature importances should not be empty"


class TestPropertyFeatureImportanceDescending:
    """プロパティベーステスト: 特徴量重要度の降順性"""
    
    # **Feature: ml-visualization-dashboard, Property 10: 特徴量重要度の降順性**
    @given(
        model_type=st.sampled_from(["random_forest", "gradient_boosting"]),
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=4, max_value=20),
        n_classes=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_feature_importances_descending_order(self, model_type, n_samples, n_features, n_classes):
        """
        プロパティ10: 任意の木ベースモデルの学習後、特徴量重要度リストが降順にソートされている
        検証: 要件 4.5
        """
        # n_informativeを計算（n_classes * n_clusters_per_class <= 2^n_informative を満たす）
        import math
        min_n_informative = max(2, math.ceil(math.log2(n_classes * 2)))
        n_informative = min(n_features, max(min_n_informative, n_features // 2))
        
        # テストデータ生成
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            random_state=42
        )
        
        # モデル作成
        hyperparameters = {
            "n_estimators": 10,  # 高速化のため少なめ
            "random_state": 42
        }
        
        model = ModelFactory.create_model(model_type, hyperparameters)
        
        # モデル学習
        trained_model, _ = ModelTrainer.train(model, X, y)
        
        # 特徴量重要度を抽出
        importances = MetricsCalculator.extract_feature_importances(trained_model)
        
        # プロパティ検証: 特徴量重要度が降順にソートされている
        assert importances is not None, f"Feature importances should not be None for {model_type}"
        assert isinstance(importances, list), "Feature importances should be a list"
        assert len(importances) == n_features, f"Expected {n_features} importances, got {len(importances)}"
        
        # 降順チェック: 各要素が次の要素以上であること
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1], \
                f"Feature importances not in descending order at index {i}: {importances[i]} < {importances[i + 1]}"
        
        # 別の方法でも検証: ソート済みリストと一致すること
        sorted_importances = sorted(importances, reverse=True)
        assert importances == sorted_importances, \
            f"Feature importances not properly sorted. Got: {importances}, Expected: {sorted_importances}"


class TestPropertyHyperparameterConsistency:
    """プロパティベーステスト: ハイパーパラメータ更新の一貫性"""
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        n_estimators=st.integers(min_value=1, max_value=200)
    )
    @settings(max_examples=100)
    def test_random_forest_n_estimators_consistency(self, n_estimators):
        """
        プロパティ6: Random Forestのn_estimatorsパラメータの一貫性
        検証: 要件 2.2
        """
        model = ModelFactory.create_model("random_forest", {"n_estimators": n_estimators})
        assert model.n_estimators == n_estimators
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        max_depth=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_random_forest_max_depth_consistency(self, max_depth):
        """
        プロパティ6: Random Forestのmax_depthパラメータの一貫性
        検証: 要件 2.3
        """
        model = ModelFactory.create_model("random_forest", {"max_depth": max_depth})
        assert model.max_depth == max_depth
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        min_samples_split=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100)
    def test_random_forest_min_samples_split_consistency(self, min_samples_split):
        """
        プロパティ6: Random Forestのmin_samples_splitパラメータの一貫性
        検証: 要件 2.4
        """
        model = ModelFactory.create_model("random_forest", {"min_samples_split": min_samples_split})
        assert model.min_samples_split == min_samples_split
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        n_estimators=st.integers(min_value=1, max_value=200),
        learning_rate=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_gradient_boosting_learning_rate_consistency(self, n_estimators, learning_rate):
        """
        プロパティ6: Gradient Boostingのlearning_rateパラメータの一貫性
        検証: 要件 2.5
        """
        model = ModelFactory.create_model("gradient_boosting", {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        })
        assert model.n_estimators == n_estimators
        assert abs(model.learning_rate - learning_rate) < 1e-6
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        C=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_svm_C_consistency(self, C):
        """
        プロパティ6: SVMのCパラメータの一貫性
        検証: 要件 2.6
        """
        model = ModelFactory.create_model("svm", {"C": C})
        assert abs(model.C - C) < 1e-6
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        C=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_logistic_regression_C_consistency(self, C):
        """
        プロパティ6: Logistic RegressionのCパラメータの一貫性
        検証: 要件 2.6
        """
        model = ModelFactory.create_model("logistic_regression", {"C": C})
        assert abs(model.C - C) < 1e-6
    
    # **Feature: ml-visualization-dashboard, Property 6: ハイパーパラメータ更新の一貫性**
    @given(
        n_neighbors=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_knn_n_neighbors_consistency(self, n_neighbors):
        """
        プロパティ6: KNNのn_neighbors (k)パラメータの一貫性
        検証: 要件 2.7
        """
        model = ModelFactory.create_model("knn", {"n_neighbors": n_neighbors})
        assert model.n_neighbors == n_neighbors



class TestPropertyModelSerializationRoundtrip:
    """プロパティベーステスト: モデルシリアライゼーションのラウンドトリップ"""
    
    # **Feature: ml-visualization-dashboard, Property 16: モデルシリアライゼーションのラウンドトリップ**
    @given(
        model_type=st.sampled_from(["random_forest", "gradient_boosting", "svm", "logistic_regression", "knn"]),
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=4, max_value=20),
        n_classes=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_model_serialization_roundtrip(self, model_type, n_samples, n_features, n_classes):
        """
        プロパティ16: 任意の学習済みモデルに対して、pickleにシリアライズしてからデシリアライズしたモデルが、
        同じテストデータに対して同じ予測結果を生成すること
        検証: 要件 6.1
        """
        # n_informativeを計算（n_classes * n_clusters_per_class <= 2^n_informative を満たす）
        import math
        min_n_informative = max(2, math.ceil(math.log2(n_classes * 2)))
        n_informative = min(n_features, max(min_n_informative, n_features // 2))
        
        # テストデータ生成
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            random_state=42
        )
        
        # データ分割（学習用とテスト用）
        split_idx = int(n_samples * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # モデル作成
        hyperparameters = {}
        if model_type in ["random_forest", "gradient_boosting"]:
            hyperparameters["n_estimators"] = 10  # 高速化のため少なめ
            hyperparameters["random_state"] = 42
        elif model_type in ["svm", "logistic_regression"]:
            hyperparameters["random_state"] = 42
            if model_type == "logistic_regression":
                hyperparameters["max_iter"] = 1000
        elif model_type == "knn":
            hyperparameters["n_neighbors"] = min(5, len(X_train) - 1)
        
        model = ModelFactory.create_model(model_type, hyperparameters)
        
        # モデル学習
        trained_model, _ = ModelTrainer.train(model, X_train, y_train)
        
        # シリアライズ前の予測
        predictions_before = trained_model.predict(X_test)
        
        # pickleにシリアライズ
        serialized = pickle.dumps(trained_model)
        
        # pickleからデシリアライズ
        deserialized_model = pickle.loads(serialized)
        
        # デシリアライズ後の予測
        predictions_after = deserialized_model.predict(X_test)
        
        # プロパティ検証: シリアライズ前後で予測結果が一致すること
        assert np.array_equal(predictions_before, predictions_after), \
            f"Predictions differ after serialization for {model_type}. " \
            f"Before: {predictions_before}, After: {predictions_after}"
        
        # 追加検証: 予測確率も一致すること（確率を返すモデルの場合）
        if hasattr(deserialized_model, 'predict_proba'):
            proba_before = trained_model.predict_proba(X_test)
            proba_after = deserialized_model.predict_proba(X_test)
            assert np.allclose(proba_before, proba_after, rtol=1e-10), \
                f"Prediction probabilities differ after serialization for {model_type}"
