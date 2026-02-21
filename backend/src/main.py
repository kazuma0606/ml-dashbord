"""
FastAPI メインアプリケーション

機械学習モデル学習・評価APIサーバー
"""
import logging
import pickle
import uuid
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .config import settings
from .models.schemas import TrainingConfig, TrainingResult, ExperimentRecord
from .models.exceptions import DatasetNotFoundError, ModelTrainingError, DatabaseError
from .services import (
    DatasetLoader,
    DataPreprocessor,
    ModelFactory,
    ModelTrainer,
    MetricsCalculator,
    get_cache_manager
)
from .repositories import get_db, init_db, ExperimentRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# インメモリモデルストレージ
trained_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時処理
    logger.info("Starting FastAPI application...")
    init_db()
    logger.info("Database initialized")
    yield
    # 終了時処理
    logger.info("Shutting down FastAPI application...")


# FastAPIアプリケーション初期化
app = FastAPI(
    title="ML Visualization Dashboard API",
    description="機械学習モデルの学習・評価・可視化API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定を環境変数から読み込み
origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(",") if settings.cors_allow_methods != "*" else ["*"],
    allow_headers=settings.cors_allow_headers.split(",") if settings.cors_allow_headers != "*" else ["*"],
)

logger.info(f"CORS configured with origins: {origins}")


# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy"}


# データセット関連エンドポイント
@app.get("/api/datasets")
async def get_datasets():
    """利用可能なデータセット一覧を取得
    
    Returns:
        List[Dict]: データセット情報のリスト
    """
    try:
        cache_manager = get_cache_manager()
        loader = DatasetLoader(cache_manager)
        datasets = loader.get_available_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Failed to get datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{name}")
async def get_dataset(name: str):
    """データセットとメタデータを取得
    
    Args:
        name: データセット名
        
    Returns:
        Dict: データセットメタデータ
    """
    try:
        cache_manager = get_cache_manager()
        loader = DatasetLoader(cache_manager)
        metadata = loader.get_dataset_metadata(name)
        return metadata
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get dataset {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{name}/preview")
async def get_dataset_preview(name: str, n_rows: int = 10):
    """データセットのプレビューを取得
    
    Args:
        name: データセット名
        n_rows: プレビュー行数（デフォルト: 10）
        
    Returns:
        Dict: プレビューデータ
    """
    try:
        cache_manager = get_cache_manager()
        loader = DatasetLoader(cache_manager)
        dataset = loader.load_dataset(name)
        preview = DataPreprocessor.prepare_preview(dataset, n_rows)
        return preview
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get dataset preview {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# モデル学習関連エンドポイント
@app.post("/api/train", response_model=TrainingResult)
async def train_model(config: TrainingConfig):
    """モデルを学習し評価指標を返す
    
    Args:
        config: 学習設定
        
    Returns:
        TrainingResult: 学習結果
    """
    try:
        logger.info(f"Training request: {config.model_dump()}")
        
        # データセット読み込み
        cache_manager = get_cache_manager()
        loader = DatasetLoader(cache_manager)
        dataset = loader.load_dataset(config.dataset_name)
        
        # データ分割
        X_train, X_test, y_train, y_test = DataPreprocessor.split_data(
            dataset.data,
            dataset.target,
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # モデル作成
        model = ModelFactory.create_model(config.model_type, config.hyperparameters)
        
        # モデル学習
        trained_model, training_time = ModelTrainer.train(model, X_train, y_train)
        
        # 予測
        y_pred = trained_model.predict(X_test)
        
        # 評価指標計算
        accuracy = MetricsCalculator.calculate_accuracy(y_test, y_pred)
        f1 = MetricsCalculator.calculate_f1_score(y_test, y_pred)
        confusion_mat = MetricsCalculator.generate_confusion_matrix(y_test, y_pred)
        classification_rep = MetricsCalculator.generate_classification_report(y_test, y_pred)
        feature_importances = MetricsCalculator.extract_feature_importances(trained_model)
        
        # モデルIDを生成してストレージに保存
        model_id = str(uuid.uuid4())
        trained_models[model_id] = {
            "model": trained_model,
            "dataset_name": config.dataset_name,
            "model_type": config.model_type,
            "config": config
        }
        
        result = TrainingResult(
            model_id=model_id,
            accuracy=accuracy,
            f1_score=f1,
            confusion_matrix=confusion_mat,
            classification_report=classification_rep,
            feature_importances=feature_importances,
            training_time=training_time
        )
        
        logger.info(f"Training completed: model_id={model_id}, accuracy={accuracy:.4f}")
        return result
        
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelTrainingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/api/models/{model_id}")
async def get_model_info(model_id: str):
    """モデル情報を取得
    
    Args:
        model_id: モデルID
        
    Returns:
        Dict: モデル情報
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model_data = trained_models[model_id]
    return {
        "model_id": model_id,
        "dataset_name": model_data["dataset_name"],
        "model_type": model_data["model_type"],
        "config": model_data["config"].model_dump()
    }


@app.get("/api/models/{model_id}/export")
async def export_model(model_id: str):
    """学習済みモデルをpickleファイルとしてエクスポート
    
    Args:
        model_id: モデルID
        
    Returns:
        Response: pickleファイル
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        model_data = trained_models[model_id]
        model = model_data["model"]
        
        # モデルをpickleにシリアライズ
        pickled_model = pickle.dumps(model)
        
        # ファイル名を生成
        filename = f"model_{model_data['model_type']}_{model_id[:8]}.pkl"
        
        return Response(
            content=pickled_model,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model export failed: {str(e)}")


# 実験履歴関連エンドポイント
@app.post("/api/experiments")
async def save_experiment(experiment: ExperimentRecord, db: Session = Depends(get_db)):
    """実験記録を保存
    
    Args:
        experiment: 実験記録
        db: データベースセッション
        
    Returns:
        Dict: 保存された実験のID
    """
    try:
        repo = ExperimentRepository(db)
        experiment_id = repo.save(experiment)
        return {"id": experiment_id, "message": "Experiment saved successfully"}
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiments")
async def get_experiments(db: Session = Depends(get_db)):
    """実験履歴を取得
    
    Args:
        db: データベースセッション
        
    Returns:
        List[ExperimentRecord]: 実験記録のリスト
    """
    try:
        repo = ExperimentRepository(db)
        experiments = repo.get_all()
        return {"experiments": experiments}
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/experiments")
async def clear_experiments(db: Session = Depends(get_db)):
    """実験履歴をクリア
    
    Args:
        db: データベースセッション
        
    Returns:
        Dict: 成功メッセージ
    """
    try:
        repo = ExperimentRepository(db)
        repo.clear()
        return {"message": "All experiments cleared successfully"}
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to clear experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# エラーハンドラー
@app.exception_handler(DatasetNotFoundError)
async def dataset_not_found_handler(request, exc: DatasetNotFoundError):
    """データセット未検出エラーハンドラー"""
    logger.error(f"Dataset not found: {exc}")
    return HTTPException(status_code=404, detail=str(exc))


@app.exception_handler(ModelTrainingError)
async def training_error_handler(request, exc: ModelTrainingError):
    """モデル学習エラーハンドラー"""
    logger.error(f"Model training error: {exc}")
    return HTTPException(status_code=500, detail=str(exc))


@app.exception_handler(DatabaseError)
async def database_error_handler(request, exc: DatabaseError):
    """データベースエラーハンドラー"""
    logger.error(f"Database error: {exc}")
    return HTTPException(status_code=503, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
