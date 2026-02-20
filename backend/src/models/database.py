"""
SQLAlchemyデータベースモデル定義
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Experiment(Base):
    """実験記録テーブル"""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now, index=True)
    dataset_name = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    accuracy = Column(Float, nullable=False, index=True)
    f1_score = Column(Float, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    training_time = Column(Float, nullable=False)
