"""
データセット読み込みモジュール

scikit-learnデータセットの読み込みとキャッシング管理
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn import datasets

from .cache import CacheManager, get_cache_manager
from ..models.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """データセット構造
    
    scikit-learnデータセットの統一インターフェース
    """
    data: np.ndarray
    target: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    DESCR: str
    name: str


class DatasetLoader:
    """データセットローダー
    
    scikit-learnデータセットの読み込みとRedisキャッシング
    """
    
    # 利用可能なデータセット定義
    AVAILABLE_DATASETS = {
        "iris": {
            "loader": datasets.load_iris,
            "description": "Iris flower dataset (3 classes, 4 features)"
        },
        "wine": {
            "loader": datasets.load_wine,
            "description": "Wine recognition dataset (3 classes, 13 features)"
        },
        "breast_cancer": {
            "loader": datasets.load_breast_cancer,
            "description": "Breast cancer wisconsin dataset (2 classes, 30 features)"
        },
        "digits": {
            "loader": datasets.load_digits,
            "description": "Digits dataset (10 classes, 64 features)"
        }
    }
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Args:
            cache_manager: キャッシュマネージャー（デフォルト: グローバルインスタンス）
        """
        self.cache_manager = cache_manager or get_cache_manager()
        self.cache_prefix = "dataset"
    
    def get_available_datasets(self) -> List[Dict[str, str]]:
        """利用可能なデータセット一覧を取得
        
        Returns:
            List[Dict[str, str]]: データセット情報のリスト
                各要素は {"name": str, "description": str} の形式
        """
        return [
            {
                "name": name,
                "description": info["description"]
            }
            for name, info in self.AVAILABLE_DATASETS.items()
        ]
    
    def load_dataset(self, name: str) -> Dataset:
        """データセットを読み込む
        
        キャッシュが存在する場合はキャッシュから取得、
        存在しない場合はscikit-learnから読み込んでキャッシュに保存
        
        Args:
            name: データセット名（例: "iris", "wine"）
            
        Returns:
            Dataset: 読み込まれたデータセット
            
        Raises:
            DatasetNotFoundError: 指定されたデータセットが存在しない
        """
        # データセット名の検証
        if name not in self.AVAILABLE_DATASETS:
            available = ", ".join(self.AVAILABLE_DATASETS.keys())
            raise DatasetNotFoundError(
                f"データセット '{name}' が見つかりません。"
                f"利用可能なデータセット: {available}"
            )
        
        # キャッシュキー生成
        cache_key = self.cache_manager.generate_cache_key(self.cache_prefix, name)
        
        # キャッシュから取得を試みる
        cached_dataset = self.cache_manager.get(cache_key)
        if cached_dataset is not None:
            logger.info(f"データセット '{name}' をキャッシュから取得")
            return cached_dataset
        
        # scikit-learnから読み込み
        logger.info(f"データセット '{name}' をscikit-learnから読み込み")
        try:
            loader_func = self.AVAILABLE_DATASETS[name]["loader"]
            sklearn_dataset = loader_func()
            
            # Dataset構造に変換
            dataset = Dataset(
                data=sklearn_dataset.data,
                target=sklearn_dataset.target,
                feature_names=list(sklearn_dataset.feature_names) if hasattr(sklearn_dataset, 'feature_names') else [f"feature_{i}" for i in range(sklearn_dataset.data.shape[1])],
                target_names=list(sklearn_dataset.target_names) if hasattr(sklearn_dataset, 'target_names') else [f"class_{i}" for i in range(len(np.unique(sklearn_dataset.target)))],
                DESCR=sklearn_dataset.DESCR if hasattr(sklearn_dataset, 'DESCR') else "",
                name=name
            )
            
            # キャッシュに保存（TTL: 1時間）
            self.cache_manager.set(cache_key, dataset, ttl=3600)
            logger.info(f"データセット '{name}' をキャッシュに保存")
            
            return dataset
            
        except Exception as e:
            logger.error(f"データセット '{name}' の読み込みエラー: {e}")
            raise DatasetNotFoundError(f"データセット '{name}' の読み込みに失敗しました: {e}")
    
    def get_dataset_metadata(self, name: str) -> Dict[str, Any]:
        """データセットのメタデータを取得
        
        Args:
            name: データセット名
            
        Returns:
            Dict[str, Any]: メタデータ（名前、サンプル数、特徴量数）
            
        Raises:
            DatasetNotFoundError: 指定されたデータセットが存在しない
        """
        dataset = self.load_dataset(name)
        
        return {
            "name": dataset.name,
            "n_samples": len(dataset.data),
            "n_features": dataset.data.shape[1],
            "n_classes": len(dataset.target_names),
            "target_names": dataset.target_names,
            "feature_names": dataset.feature_names
        }
