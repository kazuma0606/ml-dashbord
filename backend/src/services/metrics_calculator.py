"""
メトリクス計算

モデル評価指標の計算と可視化データの生成
"""
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """評価指標計算クラス"""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Accuracyを計算
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            
        Returns:
            Accuracy（0.0〜1.0）
        """
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        F1スコアを計算（加重平均）
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            
        Returns:
            F1スコア（0.0〜1.0）
        """
        return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    @staticmethod
    def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
        """
        混同行列を生成
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            
        Returns:
            混同行列（2次元リスト）
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        分類レポートを生成
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            
        Returns:
            分類レポート辞書（クラスごとのprecision、recall、F1スコア）
        """
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    @staticmethod
    def extract_feature_importances(model: BaseEstimator) -> Optional[List[float]]:
        """
        特徴量重要度を抽出（木ベースモデルのみ）
        
        Args:
            model: 学習済みモデル
            
        Returns:
            特徴量重要度リスト（降順ソート済み）、または木ベースモデルでない場合はNone
        """
        # 特徴量重要度属性を持つモデルかチェック
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # 降順にソート
            sorted_importances = sorted(importances, reverse=True)
            return [float(imp) for imp in sorted_importances]
        
        return None
