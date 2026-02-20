"""
Pydanticデータモデル定義
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """モデル学習設定"""
    dataset_name: str = Field(..., description="データセット名")
    test_size: float = Field(..., ge=0.1, le=0.5, description="テスト分割比率")
    random_state: int = Field(..., ge=0, description="乱数シード")
    model_type: str = Field(..., description="モデルタイプ")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="ハイパーパラメータ")


class TrainingResult(BaseModel):
    """モデル学習結果"""
    model_id: str = Field(..., description="モデルID")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1スコア")
    confusion_matrix: List[List[int]] = Field(..., description="混同行列")
    classification_report: Dict[str, Any] = Field(..., description="分類レポート")
    feature_importances: Optional[List[float]] = Field(None, description="特徴量重要度")
    training_time: float = Field(..., ge=0.0, description="学習時間（秒）")


class ExperimentRecord(BaseModel):
    """実験記録"""
    id: Optional[int] = Field(None, description="実験ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="実験日時")
    dataset_name: str = Field(..., description="データセット名")
    model_type: str = Field(..., description="モデルタイプ")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1スコア")
    hyperparameters: Dict[str, Any] = Field(..., description="ハイパーパラメータ")
    training_time: float = Field(..., ge=0.0, description="学習時間（秒）")
    
    class Config:
        from_attributes = True
