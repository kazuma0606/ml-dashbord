"""
モデルファクトリー

モデルタイプに基づいてscikit-learnモデルをインスタンス化
"""
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

from ..models.exceptions import ModelTrainingError


class ModelFactory:
    """モデルインスタンス化ファクトリー"""
    
    # サポートされるモデルタイプ
    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svm": SVC,
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
    }
    
    @staticmethod
    def create_model(model_type: str, hyperparameters: Dict[str, Any]) -> BaseEstimator:
        """
        モデルタイプとハイパーパラメータからモデルインスタンスを作成
        
        Args:
            model_type: モデルタイプ（random_forest, gradient_boosting, svm, logistic_regression, knn）
            hyperparameters: ハイパーパラメータ辞書
            
        Returns:
            インスタンス化されたscikit-learnモデル
            
        Raises:
            ModelTrainingError: サポートされていないモデルタイプの場合
        """
        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ModelTrainingError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(ModelFactory.SUPPORTED_MODELS.keys())}"
            )
        
        model_class = ModelFactory.SUPPORTED_MODELS[model_type]
        
        # モデルタイプに応じて適切なハイパーパラメータをフィルタリング
        filtered_params = ModelFactory._filter_hyperparameters(model_type, hyperparameters)
        
        try:
            model = model_class(**filtered_params)
            return model
        except Exception as e:
            raise ModelTrainingError(f"Failed to create model: {str(e)}")
    
    @staticmethod
    def _filter_hyperparameters(model_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        モデルタイプに応じて適切なハイパーパラメータのみを抽出
        
        Args:
            model_type: モデルタイプ
            hyperparameters: 全ハイパーパラメータ
            
        Returns:
            フィルタリングされたハイパーパラメータ
        """
        # 各モデルタイプで有効なパラメータを定義
        valid_params = {
            "random_forest": {"n_estimators", "max_depth", "min_samples_split", "random_state"},
            "gradient_boosting": {"n_estimators", "max_depth", "min_samples_split", "learning_rate", "random_state"},
            "svm": {"C", "kernel", "random_state"},
            "logistic_regression": {"C", "max_iter", "random_state"},
            "knn": {"n_neighbors"},
        }
        
        allowed_params = valid_params.get(model_type, set())
        
        # 有効なパラメータのみを抽出
        filtered = {k: v for k, v in hyperparameters.items() if k in allowed_params}
        
        return filtered
