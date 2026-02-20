# ML Dashboard

機械学習モデルの学習・評価・可視化を行うWebダッシュボードシステム（開発中）

## 概要

scikit-learnのデータセットを用いて、複数の機械学習モデルを学習・評価し、結果を可視化するダッシュボードアプリケーションです。

## 技術スタック

- **フロントエンド**: Streamlit
- **バックエンド**: FastAPI
- **データベース**: PostgreSQL
- **キャッシュ**: Redis
- **パッケージ管理**: uv
- **コンテナ**: Docker / Docker Compose

## プロジェクト構成

```
ml-visualization-dashboard/
├── backend/          # FastAPI バックエンド
├── frontend/         # Streamlit フロントエンド
└── .kiro/specs/      # 仕様書
```

## セットアップ

### 前提条件

- Python 3.10以上
- uv (Pythonパッケージマネージャー)
- Docker & Docker Compose (本番環境用)

### uvのインストール

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 依存関係のインストール

プロジェクトルートで以下を実行：

```bash
# すべての依存関係をインストール（モノレポ全体）
uv sync

# バックエンドのみ
cd backend
uv sync

# フロントエンドのみ
cd frontend
uv sync
```

### Docker環境での起動（推奨）

```bash
# 環境変数ファイルを作成
cp .env.example .env

# すべてのサービスを起動
docker-compose up -d

# ログを確認
docker-compose logs -f

# サービスを停止
docker-compose down

# データも削除する場合
docker-compose down -v
```

起動後、以下のURLでアクセス可能：
- フロントエンド: http://localhost:8501
- バックエンドAPI: http://localhost:8000
- API ドキュメント: http://localhost:8000/docs

### ローカル開発環境の起動

```bash
# 環境変数ファイルを作成
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# バックエンド（FastAPI）
cd backend
uv run uvicorn src.main:app --reload

# フロントエンド（Streamlit）
cd frontend
uv run streamlit run src/app.py
```

注意: ローカル開発時は、PostgreSQLとRedisを別途起動する必要があります。

## 開発状況

- ✅ プロジェクト構造とuv環境のセットアップ
- ✅ Docker環境のセットアップ

詳細な仕様は `.kiro/specs/ml-visualization-dashboard/` を参照してください：
- `requirements.md` - 要件定義書
- `design.md` - 設計書
- `tasks.md` - 実装計画

## ライセンス

MIT
