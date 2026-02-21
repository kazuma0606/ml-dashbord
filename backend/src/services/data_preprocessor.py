"""
データ前処理モジュール

データ分割、プレビュー生成などの前処理機能
"""
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_loader import Dataset

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """データ前処理クラス
    
    データ分割、プレビュー生成などの前処理機能を提供
    """
    
    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """データをtrain/testに分割
        
        Args:
            X: 特徴量データ
            y: ターゲットデータ
            test_size: テストセットの割合（0.1〜0.5）
            random_state: 乱数シード（再現性のため）
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (X_train, X_test, y_train, y_test)
        """
        logger.info(
            f"データ分割: test_size={test_size}, random_state={random_state}"
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # クラス比率を保持
        )
        
        logger.info(
            f"分割完了: train={len(X_train)}, test={len(X_test)}"
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def prepare_preview(
        dataset: Dataset,
        n_rows: int = 10
    ) -> Dict[str, Any]:
        """データプレビューを生成
        
        Args:
            dataset: データセット
            n_rows: プレビュー行数（デフォルト: 10）
            
        Returns:
            Dict[str, Any]: プレビューデータ
                {
                    "data": List[Dict]: 各行のデータ（特徴量 + ターゲット）
                    "columns": List[str]: カラム名リスト
                    "n_rows": int: 実際の行数
                }
        """
        # 実際のプレビュー行数を決定（データセットサイズを超えない）
        actual_n_rows = min(n_rows, len(dataset.data))
        
        # DataFrameを作成
        df = pd.DataFrame(
            dataset.data[:actual_n_rows],
            columns=dataset.feature_names
        )
        
        # ターゲット列を追加（ラベル名に変換）
        target_labels = [
            dataset.target_names[int(t)] if int(t) < len(dataset.target_names) else f"class_{int(t)}"
            for t in dataset.target[:actual_n_rows]
        ]
        df["target"] = target_labels
        
        # 辞書形式に変換
        preview_data = df.to_dict(orient="records")
        columns = list(df.columns)
        
        logger.info(
            f"プレビュー生成: {actual_n_rows}行 x {len(columns)}列"
        )
        
        return {
            "data": preview_data,
            "columns": columns,
            "n_rows": actual_n_rows
        }
    
    @staticmethod
    def get_split_info(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """分割情報を取得
        
        Args:
            X_train: 訓練特徴量
            X_test: テスト特徴量
            y_train: 訓練ターゲット
            y_test: テストターゲット
            
        Returns:
            Dict[str, Any]: 分割情報
        """
        total_samples = len(X_train) + len(X_test)
        actual_test_ratio = len(X_test) / total_samples if total_samples > 0 else 0
        
        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": total_samples,
            "actual_test_ratio": actual_test_ratio,
            "n_features": X_train.shape[1] if len(X_train) > 0 else 0
        }
