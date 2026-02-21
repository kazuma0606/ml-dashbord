"""
モデルトレーナー

モデルの学習と評価を実行
"""
import time
import logging
from typing import Tuple
import numpy as np
from sklearn.base import BaseEstimator

from ..models.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class ModelTrainer:
    """モデル学習クラス"""
    
    @staticmethod
    def train(
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[BaseEstimator, float]:
        """
        モデルを学習し、学習時間を計測
        
        Args:
            model: 学習するモデル
            X_train: 学習データ（特徴量）
            y_train: 学習データ（ターゲット）
            
        Returns:
            (学習済みモデル, 学習時間（秒）)
            
        Raises:
            ModelTrainingError: 学習中にエラーが発生した場合
        """
        try:
            logger.info(f"Starting model training: {type(model).__name__}")
            
            # 学習時間計測開始
            start_time = time.time()
            
            # モデル学習
            model.fit(X_train, y_train)
            
            # 学習時間計測終了
            training_time = time.time() - start_time
            
            logger.info(
                f"Model training completed: {type(model).__name__}, "
                f"training_time={training_time:.3f}s"
            )
            
            return model, training_time
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            raise ModelTrainingError(error_msg)
