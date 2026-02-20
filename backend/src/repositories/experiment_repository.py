"""
実験記録リポジトリ
"""
import logging
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..models.database import Experiment
from ..models.schemas import ExperimentRecord
from ..models.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class ExperimentRepository:
    """実験記録のデータアクセスクラス"""
    
    def __init__(self, db: Session):
        """
        Args:
            db: SQLAlchemyセッション
        """
        self.db = db
    
    def save(self, experiment: ExperimentRecord) -> int:
        """
        実験記録を保存
        
        Args:
            experiment: 実験記録
            
        Returns:
            int: 保存された実験のID
            
        Raises:
            DatabaseError: データベース操作に失敗した場合
        """
        try:
            db_experiment = Experiment(
                timestamp=experiment.timestamp,
                dataset_name=experiment.dataset_name,
                model_type=experiment.model_type,
                accuracy=experiment.accuracy,
                f1_score=experiment.f1_score,
                hyperparameters=experiment.hyperparameters,
                training_time=experiment.training_time,
            )
            self.db.add(db_experiment)
            self.db.commit()
            self.db.refresh(db_experiment)
            
            logger.info(f"実験記録を保存しました（ID: {db_experiment.id}）")
            return db_experiment.id
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"実験記録の保存に失敗しました: {e}")
            raise DatabaseError(f"実験記録の保存に失敗しました: {e}")
    
    def get_all(self) -> List[ExperimentRecord]:
        """
        すべての実験記録を取得（時系列降順）
        
        Returns:
            List[ExperimentRecord]: 実験記録のリスト
            
        Raises:
            DatabaseError: データベース操作に失敗した場合
        """
        try:
            experiments = (
                self.db.query(Experiment)
                .order_by(Experiment.timestamp.desc())
                .all()
            )
            
            return [
                ExperimentRecord(
                    id=exp.id,
                    timestamp=exp.timestamp,
                    dataset_name=exp.dataset_name,
                    model_type=exp.model_type,
                    accuracy=exp.accuracy,
                    f1_score=exp.f1_score,
                    hyperparameters=exp.hyperparameters,
                    training_time=exp.training_time,
                )
                for exp in experiments
            ]
            
        except SQLAlchemyError as e:
            logger.error(f"実験記録の取得に失敗しました: {e}")
            raise DatabaseError(f"実験記録の取得に失敗しました: {e}")
    
    def clear(self) -> bool:
        """
        すべての実験記録を削除
        
        Returns:
            bool: 削除成功時True
            
        Raises:
            DatabaseError: データベース操作に失敗した場合
        """
        try:
            deleted_count = self.db.query(Experiment).delete()
            self.db.commit()
            
            logger.info(f"{deleted_count}件の実験記録を削除しました")
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"実験記録の削除に失敗しました: {e}")
            raise DatabaseError(f"実験記録の削除に失敗しました: {e}")
