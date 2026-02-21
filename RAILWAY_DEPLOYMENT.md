# Railway デプロイメントガイド（更新版）

このガイドでは、RailwayにML Dashboardをデプロイする手順を説明します。外部データベース（SupabaseやUpstash）を使用する方法も含まれています。

## 前提条件

- Railwayアカウント
- GitHubリポジトリにプロジェクトがプッシュされていること
- （オプション）Supabase PostgreSQLアカウント
- （オプション）Upstash Redisアカウント

## デプロイ方法

### 方法1: 外部データベースを使用（推奨 - 無料枠が大きい）

SupabaseとUpstashを使用すると、Railwayの無料枠を節約できます。

#### ステップ1: Supabase PostgreSQLのセットアップ

1. [Supabase](https://supabase.com/)にサインアップ
2. 新しいプロジェクトを作成
3. Project Settings → Database → Connection stringから接続URLを取得
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xxx.supabase.co:5432/postgres
   ```

#### ステップ2: Upstash Redisのセットアップ

1. [Upstash](https://upstash.com/)にサインアップ
2. 新しいRedisデータベースを作成
3. Detailsタブから接続URLを取得
   ```
   redis://default:[YOUR-PASSWORD]@xxx.upstash.io:6379
   ```

#### ステップ3: Railwayでバックエンドをデプロイ

1. Railwayダッシュボードで「New Project」をクリック
2. 「Deploy from GitHub repo」を選択
3. リポジトリを選択
4. サービス設定：
   - **Root Directory**: `backend`
   - **環境変数**:
     ```
     DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.xxx.supabase.co:5432/postgres
     REDIS_URL=redis://default:[YOUR-PASSWORD]@xxx.upstash.io:6379
     CORS_ORIGINS=*
     ```
5. 「Deploy」をクリック

#### ステップ4: Railwayでフロントエンドをデプロイ

1. 同じプロジェクトで「New」→「GitHub Repo」を選択
2. 同じリポジトリを選択
3. サービス設定：
   - **Root Directory**: `frontend`
   - **環境変数**:
     ```
     API_BASE_URL=https://[backend-domain].up.railway.app
     ```
4. 「Deploy」をクリック

### 方法2: Railway内部データベースを使用

#### ステップ1: PostgreSQLとRedisを追加

1. Railwayプロジェクトで「New」→「Database」→「Add PostgreSQL」
2. 「New」→「Database」→「Add Redis」

#### ステップ2: バックエンドをデプロイ

1. 「New」→「GitHub Repo」を選択
2. サービス設定：
   - **Root Directory**: `backend`
   - **環境変数**:
     ```
     POSTGRES_HOST=${{Postgres.PGHOST}}
     POSTGRES_PORT=${{Postgres.PGPORT}}
     POSTGRES_USER=${{Postgres.PGUSER}}
     POSTGRES_PASSWORD=${{Postgres.PGPASSWORD}}
     POSTGRES_DB=${{Postgres.PGDATABASE}}
     REDIS_HOST=${{Redis.REDIS_HOST}}
     REDIS_PORT=${{Redis.REDIS_PORT}}
     REDIS_DB=0
     CORS_ORIGINS=*
     ```

#### ステップ3: フロントエンドをデプロイ

1. 「New」→「GitHub Repo」を選択
2. サービス設定：
   - **Root Directory**: `frontend`
   - **環境変数**:
     ```
     API_BASE_URL=https://[backend-domain].up.railway.app
     ```

## 環境変数の詳細

### Backend環境変数

**オプション1: DATABASE_URLとREDIS_URLを使用（推奨 - SupabaseやUpstashの場合）**
```bash
DATABASE_URL=postgresql://postgres:password@host:5432/database
REDIS_URL=redis://default:password@host:6379
CORS_ORIGINS=*
```

**オプション2: 個別の環境変数を使用（Railway内部データベースの場合）**
```bash
POSTGRES_HOST=${{Postgres.PGHOST}}
POSTGRES_PORT=${{Postgres.PGPORT}}
POSTGRES_USER=${{Postgres.PGUSER}}
POSTGRES_PASSWORD=${{Postgres.PGPASSWORD}}
POSTGRES_DB=${{Postgres.PGDATABASE}}
REDIS_HOST=${{Redis.REDIS_HOST}}
REDIS_PORT=${{Redis.REDIS_PORT}}
REDIS_DB=0
CORS_ORIGINS=*
```

### Frontend環境変数

```bash
API_BASE_URL=https://[backend-domain].up.railway.app
API_TIMEOUT=30
APP_TITLE=ML Dashboard
APP_ICON=🤖
```

## トラブルシューティング

### エラー: "connection to server at localhost"

これは環境変数が正しく設定されていないことを示しています。

**解決策:**
1. Railwayのサービス設定で環境変数を確認
2. `DATABASE_URL`または個別のPostgreSQL設定が正しいか確認
3. サービスを再デプロイ

### エラー: "No start command was found"

これはモノレポ構成の問題です。

**解決策:**
1. Root Directoryが`backend`または`frontend`に設定されているか確認
2. `nixpacks.toml`または`Procfile`が各ディレクトリに存在するか確認

### CORS エラー

**解決策:**
1. Backend環境変数で`CORS_ORIGINS`を確認
2. Frontend のドメインを含めるか、`*`を設定

## コスト最適化

### 無料枠の活用

- **Supabase**: 500MB データベース、無制限API リクエスト（無料）
- **Upstash**: 10,000コマンド/日（無料）
- **Railway**: $5/月のクレジット（無料）

外部データベースを使用することで、Railwayの無料枠をアプリケーションのみに使用できます。

## まとめ

Railwayを使用すると、簡単にアプリケーションをデプロイできます。

アクセスURL:
- Frontend: `https://[frontend-domain].up.railway.app`
- Backend API: `https://[backend-domain].up.railway.app/docs`

問題が発生した場合は、Railwayのログを確認してください。
