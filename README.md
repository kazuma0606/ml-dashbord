# ML Dashboard

機械学習モデルの学習・評価・可視化を行うWebダッシュボードシステム

## 概要

scikit-learnのデータセットを用いて、複数の機械学習モデルを学習・評価し、結果を可視化するダッシュボードアプリケーションです。

## 主な機能

- 📊 **データセット管理**: Iris, Wine, Breast Cancer, Digitsなどのscikit-learn組み込みデータセット
- 🤖 **モデル選択**: Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN
- ⚙️ **ハイパーパラメータ調整**: モデルごとの動的パラメータ設定
- 📈 **リアルタイム可視化**: 混同行列、特徴量重要度、分類レポート
- 💾 **実験履歴管理**: パラメータと結果の永続化・比較
- 🚀 **モデルエクスポート**: 学習済みモデルのpickle形式ダウンロード

## 技術スタック

- **フロントエンド**: Streamlit
- **バックエンド**: FastAPI
- **データベース**: PostgreSQL
- **キャッシュ**: Redis
- **パッケージ管理**: uv
- **コンテナ**: Docker / Docker Compose
- **テスト**: pytest, Hypothesis (Property-Based Testing)

## プロジェクト構成

```
ml-visualization-dashboard/
├── backend/              # FastAPI バックエンド
│   ├── src/
│   │   ├── main.py      # FastAPIアプリケーション
│   │   ├── config.py    # 環境変数設定
│   │   ├── models/      # Pydanticモデル
│   │   ├── services/    # ビジネスロジック
│   │   └── repositories/# データアクセス
│   ├── tests/           # テストスイート
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/            # Streamlit フロントエンド
│   ├── src/
│   │   ├── app.py       # メインアプリケーション
│   │   ├── config.py    # 環境変数設定
│   │   ├── api_client.py# APIクライアント
│   │   └── components/  # UIコンポーネント
│   ├── tests/           # テストスイート
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml   # Docker Compose設定
├── .env.example         # 環境変数テンプレート
└── .kiro/specs/         # 仕様書
```

## セットアップ

### 前提条件

- Python 3.10以上
- uv (Pythonパッケージマネージャー)
- Docker & Docker Compose (推奨)

### uvのインストール

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、ターミナルを再起動してuvコマンドが使用可能か確認：

```bash
uv --version
```

### クイックスタート（Docker）

最も簡単な起動方法：

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd ml-visualization-dashboard

# 2. 環境変数ファイルを作成
cp .env.example .env

# 3. すべてのサービスを起動
docker-compose up -d

# 4. ログを確認（オプション）
docker-compose logs -f
```

起動後、ブラウザで以下にアクセス：
- **フロントエンド**: http://localhost:8501
- **バックエンドAPI**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs

### Docker環境の詳細操作

```bash
# サービスの状態確認
docker-compose ps

# 特定のサービスのログを確認
docker-compose logs -f backend
docker-compose logs -f frontend

# サービスを再起動
docker-compose restart

# サービスを停止
docker-compose down

# データボリュームも削除（データベースとキャッシュをリセット）
docker-compose down -v

# イメージを再ビルド
docker-compose build

# 再ビルドして起動
docker-compose up -d --build
```

### ローカル開発環境のセットアップ

Docker を使わずにローカルで開発する場合：

#### 1. 依存関係のインストール

```bash
# プロジェクトルートで実行
uv sync

# または個別にインストール
cd backend && uv sync
cd ../frontend && uv sync
```

#### 2. PostgreSQLとRedisの起動

```bash
# Docker Composeで必要なサービスのみ起動
docker-compose up -d postgres redis
```

または、ローカルにインストールされたPostgreSQLとRedisを使用。

#### 3. 環境変数の設定

```bash
# バックエンド
cp backend/.env.example backend/.env
# backend/.envを編集してローカル設定に変更

# フロントエンド
cp frontend/.env.example frontend/.env
# frontend/.envを編集してローカル設定に変更
```

#### 4. サービスの起動

```bash
# ターミナル1: バックエンド
cd backend
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# ターミナル2: フロントエンド
cd frontend
uv run streamlit run src/app.py
```

## 使用方法

### 基本的なワークフロー

1. **データセット選択**
   - サイドバーからデータセット（Iris, Wine, Breast Cancer, Digits）を選択
   - テスト分割比率と乱数シードを設定

2. **モデル設定**
   - モデルタイプを選択（Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN）
   - ハイパーパラメータをスライダーで調整

3. **学習実行**
   - 「学習開始」ボタンをクリック
   - プログレスバーで進捗を確認

4. **結果確認**
   - Accuracy、F1スコアなどのメトリクスを確認
   - 混同行列、特徴量重要度、分類レポートを可視化
   - データプレビューで入力データを確認

5. **実験管理**
   - 「パラメータ保存」で現在の設定を保存
   - 実験履歴テーブルで過去の実行結果を比較
   - Accuracy推移グラフで最良モデルを確認

6. **モデルエクスポート**
   - 「モデル出力」ボタンで学習済みモデルをダウンロード
   - pickleファイルとして保存され、本番環境で使用可能

### 環境変数の設定

`.env`ファイルで以下の設定をカスタマイズ可能：

```bash
# データベース設定
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ml_dashboard

# CORS設定（フロントエンドのURLを追加）
CORS_ORIGINS=http://localhost:8501,http://frontend:8501

# バックエンドAPI URL（フロントエンドから接続）
API_BASE_URL=http://backend:8000

# ポート設定
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

## テスト

### バックエンドテスト

```bash
cd backend

# すべてのテストを実行
uv run pytest

# カバレッジ付きで実行
uv run pytest --cov=src --cov-report=html

# 特定のテストファイルを実行
uv run pytest tests/test_api_endpoints.py

# プロパティベーステストのみ実行
uv run pytest tests/test_property_*.py -v
```

### フロントエンドテスト

```bash
cd frontend

# すべてのテストを実行
uv run pytest

# プロパティベーステストのみ実行
uv run pytest tests/test_property_*.py -v
```

### 統合テスト

```bash
cd backend

# 統合テスト（testcontainersを使用）
uv run pytest tests/test_integration_*.py -v
```

## トラブルシューティング

### Docker関連

**問題**: コンテナが起動しない

```bash
# ログを確認
docker-compose logs

# 特定のサービスのログを確認
docker-compose logs backend
docker-compose logs postgres

# コンテナの状態を確認
docker-compose ps
```

**問題**: ポートが既に使用されている

`.env`ファイルでポート番号を変更：

```bash
BACKEND_PORT=8001
FRONTEND_PORT=8502
POSTGRES_PORT=5433
REDIS_PORT=6380
```

**問題**: データベース接続エラー

```bash
# PostgreSQLが起動しているか確認
docker-compose ps postgres

# ヘルスチェックの状態を確認
docker-compose ps

# データベースをリセット
docker-compose down -v
docker-compose up -d
```

### ローカル開発関連

**問題**: uvコマンドが見つからない

```bash
# uvを再インストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# パスを確認
echo $PATH

# ターミナルを再起動
```

**問題**: 依存関係のインストールエラー

```bash
# キャッシュをクリア
uv cache clean

# 再インストール
uv sync --reinstall
```

**問題**: API接続エラー

- バックエンドが起動しているか確認
- `frontend/.env`の`API_BASE_URL`が正しいか確認
- CORS設定が正しいか確認（`backend/.env`の`CORS_ORIGINS`）

### パフォーマンス関連

**問題**: 学習が遅い

- データセットサイズを確認
- ハイパーパラメータを調整（n_estimatorsを減らすなど）
- Redisキャッシュが有効か確認

**問題**: メモリ不足

- Docker Desktopのメモリ割り当てを増やす
- 大きなモデル（n_estimators、max_depth）を避ける

## 開発ガイド

### コードスタイル

```bash
# バックエンド
cd backend
uv run ruff check src/
uv run ruff format src/

# フロントエンド
cd frontend
uv run ruff check src/
uv run ruff format src/
```

### 新しい依存関係の追加

```bash
# バックエンドに追加
cd backend
uv add <package-name>

# フロントエンドに追加
cd frontend
uv add <package-name>

# 開発依存関係として追加
uv add --dev <package-name>
```

### データベースマイグレーション

現在のバージョンではマイグレーションツールは使用していませんが、スキーマ変更時は：

```bash
# データベースをリセット
docker-compose down -v
docker-compose up -d postgres

# バックエンドを再起動（自動的にテーブルを作成）
docker-compose restart backend
```

## アーキテクチャ

システムは4つの主要コンポーネントで構成：

1. **Streamlit Frontend**: ユーザーインターフェース
2. **FastAPI Backend**: RESTful API、ビジネスロジック
3. **PostgreSQL**: 実験履歴の永続化
4. **Redis**: データセットのキャッシング

詳細な設計は `.kiro/specs/ml-visualization-dashboard/design.md` を参照。

## 仕様書

プロジェクトの詳細な仕様は `.kiro/specs/ml-visualization-dashboard/` に格納：

- `requirements.md` - 要件定義書（EARS形式）
- `design.md` - 設計書（正確性プロパティ含む）
- `tasks.md` - 実装計画

## ライセンス

MIT
