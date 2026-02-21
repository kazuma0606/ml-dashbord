# Fly.io デプロイメントガイド

このガイドでは、Fly.ioにML Dashboardをデプロイする手順を説明します。

## 前提条件

- Fly.ioアカウント（無料枠あり）
- flyctlコマンドラインツール
- GitHubリポジトリ

## Fly.ioの特徴

- ✅ Dockerfileベースのデプロイ
- ✅ 自動スケーリング
- ✅ グローバルエッジネットワーク
- ✅ PostgreSQLとRedisのマネージドサービス
- ✅ 無料枠: 3つのVMまで無料

## セットアップ

### 1. Fly.io CLIのインストール

#### Windows (PowerShell)
```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

#### macOS/Linux
```bash
curl -L https://fly.io/install.sh | sh
```

インストール後、パスを通す：
```bash
# ~/.bashrc または ~/.zshrc に追加
export FLYCTL_INSTALL="/home/username/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
```

### 2. Fly.ioにログイン

```bash
flyctl auth login
```

ブラウザが開き、ログインページが表示されます。

### 3. Fly.ioアカウントの確認

```bash
flyctl auth whoami
```

## デプロイ手順

### ステップ1: PostgreSQLデータベースの作成

```bash
# PostgreSQLクラスターを作成
flyctl postgres create --name ml-dashboard-db --region nrt

# 接続情報を確認
flyctl postgres connect -a ml-dashboard-db
```

作成後、接続文字列が表示されます。これを後で使用します。

### ステップ2: Redisの作成

```bash
# Redisインスタンスを作成
flyctl redis create --name ml-dashboard-redis --region nrt

# 接続情報を確認
flyctl redis status ml-dashboard-redis
```

### ステップ3: Backendのデプロイ

```bash
# backendディレクトリに移動
cd backend

# Fly.ioアプリを作成（初回のみ）
flyctl launch --no-deploy

# アプリ名を確認（例: ml-dashboard-backend）
# fly.tomlが自動生成されますが、既に用意したものを使用します

# シークレット（環境変数）を設定
flyctl secrets set \
  POSTGRES_HOST=ml-dashboard-db.internal \
  POSTGRES_PORT=5432 \
  POSTGRES_USER=postgres \
  POSTGRES_PASSWORD=<YOUR-DB-PASSWORD> \
  POSTGRES_DB=ml_dashboard \
  REDIS_HOST=ml-dashboard-redis.internal \
  REDIS_PORT=6379 \
  REDIS_DB=0

# デプロイ
flyctl deploy

# デプロイ状況を確認
flyctl status

# ログを確認
flyctl logs
```

### ステップ4: Frontendのデプロイ

```bash
# ルートディレクトリに戻る
cd ..

# frontendディレクトリに移動
cd frontend

# Fly.ioアプリを作成（初回のみ）
flyctl launch --no-deploy

# BackendのURLを取得
cd ../backend
flyctl info

# FrontendにBackend URLを設定
cd ../frontend
flyctl secrets set API_BASE_URL=https://<backend-app-name>.fly.dev

# デプロイ
flyctl deploy

# デプロイ状況を確認
flyctl status

# ログを確認
flyctl logs
```

### ステップ5: アプリケーションへのアクセス

```bash
# Frontendを開く
cd frontend
flyctl open

# または直接URLにアクセス
# https://<frontend-app-name>.fly.dev
```

## 環境変数の設定

### Backend環境変数

```bash
cd backend

# データベース接続
flyctl secrets set POSTGRES_HOST=ml-dashboard-db.internal
flyctl secrets set POSTGRES_PORT=5432
flyctl secrets set POSTGRES_USER=postgres
flyctl secrets set POSTGRES_PASSWORD=<YOUR-PASSWORD>
flyctl secrets set POSTGRES_DB=ml_dashboard

# Redis接続
flyctl secrets set REDIS_HOST=ml-dashboard-redis.internal
flyctl secrets set REDIS_PORT=6379
flyctl secrets set REDIS_DB=0

# CORS設定
flyctl secrets set CORS_ORIGINS=https://<frontend-app-name>.fly.dev

# 環境変数の確認
flyctl secrets list
```

### Frontend環境変数

```bash
cd frontend

# Backend API URL
flyctl secrets set API_BASE_URL=https://<backend-app-name>.fly.dev

# 環境変数の確認
flyctl secrets list
```

## 管理コマンド

### アプリケーションの管理

```bash
# アプリの状態確認
flyctl status

# ログの確認
flyctl logs

# リアルタイムログ
flyctl logs -f

# アプリを開く
flyctl open

# アプリの再起動
flyctl apps restart <app-name>

# スケーリング
flyctl scale count 2  # 2つのインスタンスに増やす
flyctl scale vm shared-cpu-1x --memory 512  # VMサイズを変更
```

### データベースの管理

```bash
# PostgreSQL接続
flyctl postgres connect -a ml-dashboard-db

# データベースのバックアップ
flyctl postgres backup create -a ml-dashboard-db

# バックアップ一覧
flyctl postgres backup list -a ml-dashboard-db
```

### アプリケーションの更新

```bash
# コードを更新
git pull

# Backendを再デプロイ
cd backend
flyctl deploy

# Frontendを再デプロイ
cd ../frontend
flyctl deploy
```

### アプリケーションの削除

```bash
# アプリを削除
flyctl apps destroy <app-name>

# データベースを削除
flyctl postgres destroy ml-dashboard-db

# Redisを削除
flyctl redis destroy ml-dashboard-redis
```

## トラブルシューティング

### 問題: デプロイが失敗する

```bash
# ログを確認
flyctl logs

# ビルドログを確認
flyctl deploy --verbose

# アプリの状態を確認
flyctl status
```

### 問題: データベースに接続できない

```bash
# データベースの状態を確認
flyctl postgres status ml-dashboard-db

# 接続情報を確認
flyctl postgres connect -a ml-dashboard-db

# 環境変数を確認
flyctl secrets list
```

### 問題: Frontendがバックendに接続できない

1. Backend URLが正しいか確認
   ```bash
   cd backend
   flyctl info
   ```

2. Frontend環境変数を確認
   ```bash
   cd frontend
   flyctl secrets list
   ```

3. CORS設定を確認
   ```bash
   cd backend
   flyctl secrets list | grep CORS
   ```

### 問題: メモリ不足

```bash
# VMサイズを増やす
flyctl scale vm shared-cpu-1x --memory 1024

# または
flyctl scale vm shared-cpu-2x --memory 2048
```

## コスト最適化

### 無料枠の活用

Fly.ioの無料枠：
- 3つのshared-cpu-1x VM（256MB RAM）
- 3GB永続ストレージ
- 160GB転送量/月

### 自動停止の設定

`fly.toml`で自動停止を有効化（既に設定済み）：
```toml
[http_service]
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
```

これにより、アクセスがない時は自動的に停止し、アクセスがあると自動起動します。

### リソース監視

```bash
# リソース使用状況を確認
flyctl status
flyctl metrics

# ログでエラーを監視
flyctl logs -f
```

## セキュリティ

### シークレットの管理

```bash
# シークレットを設定（環境変数として保存）
flyctl secrets set KEY=VALUE

# シークレット一覧（値は表示されない）
flyctl secrets list

# シークレットを削除
flyctl secrets unset KEY
```

### HTTPSの強制

`fly.toml`で既に設定済み：
```toml
[http_service]
  force_https = true
```

### プライベートネットワーク

PostgreSQLとRedisは`.internal`ドメインでプライベートネットワーク経由で接続されます。

## モニタリング

### ログの確認

```bash
# リアルタイムログ
flyctl logs -f

# 特定のアプリのログ
flyctl logs -a ml-dashboard-backend

# エラーログのみ
flyctl logs | grep ERROR
```

### メトリクス

```bash
# メトリクスを確認
flyctl metrics

# ダッシュボードを開く
flyctl dashboard
```

## 高度な設定

### カスタムドメインの設定

```bash
# カスタムドメインを追加
flyctl certs add your-domain.com

# SSL証明書の状態を確認
flyctl certs show your-domain.com
```

### 複数リージョンへのデプロイ

```bash
# 別のリージョンにスケール
flyctl regions add sin  # シンガポール
flyctl regions add syd  # シドニー

# リージョン一覧
flyctl regions list
```

## クイックリファレンス

### よく使うコマンド

```bash
# デプロイ
flyctl deploy

# ログ確認
flyctl logs -f

# アプリを開く
flyctl open

# 状態確認
flyctl status

# 環境変数設定
flyctl secrets set KEY=VALUE

# スケーリング
flyctl scale count 2

# アプリ削除
flyctl apps destroy <app-name>
```

### 接続情報の確認

```bash
# Backend URL
cd backend && flyctl info

# Frontend URL
cd frontend && flyctl info

# Database接続
flyctl postgres connect -a ml-dashboard-db

# Redis接続
flyctl redis status ml-dashboard-redis
```

## まとめ

Fly.ioを使用すると、グローバルエッジネットワークで高速なアプリケーションをデプロイできます。

主な利点：
- ✅ 自動スケーリング
- ✅ グローバル展開
- ✅ 無料枠が充実
- ✅ Dockerfileベースで簡単

アクセスURL:
- Frontend: `https://<frontend-app-name>.fly.dev`
- Backend API: `https://<backend-app-name>.fly.dev/docs`

問題が発生した場合は、ログを確認してください：
```bash
flyctl logs -f
```
