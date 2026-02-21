# ML Dashboard — デプロイ作業記録

**日付**: 2026年2月21日  
**プラットフォーム**: Railway + Supabase + Upstash

## 1. プロジェクト概要

本ドキュメントは、ML Visualization Dashboard のデプロイ作業において発生したエラーとその解決策を時系列で記録したものです。

### 1-1. 技術スタック

| 役割 | 技術 | 備考 |
|------|------|------|
| フロントエンド | Streamlit | 機械学習UI・可視化 |
| バックエンド | FastAPI | REST API / ML推論エンジン |
| データベース | PostgreSQL (Supabase) | 実験履歴の永続化 |
| キャッシュ | Redis (Upstash) | モデルキャッシュ |
| パッケージ管理 | uv → pip | デプロイ時に切り替え |
| コンテナ | Docker / Docker Compose | ローカル開発環境 |
| デプロイ先 | Railway | PaaS (モノレポ対応) |

### 1-2. リポジトリ構成

```
ml-dashboard/              ← モノレポ
├── backend/              ← FastAPI
│   ├── src/
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/             ← Streamlit
│   └── src/
├── docker-compose.yml
└── .env.example
```

## 2. デプロイ方針

当初は `docker-compose.yml` をそのまま Railway に乗せることを検討したが、モノレポ構成との相性問題により以下の方針に変更した。

### 採用した方針

- Railway でリポジトリを `backend` / `frontend` の 2サービスに分けてデプロイ
- PostgreSQL → Supabase (BaaS) に変更
- Redis → Upstash (BaaS) に変更
- uv → pip + requirements.txt に切り替え（Railway 環境への対応）

## 3. エラーログ & トラブルシューティング

デプロイ作業中に発生したエラーを発生順に記録する。

### エラー 1：No start command was found

**原因**  
Nixpacks がモノレポのルートを見てアプリを検出できなかった。ルートに `main.py` や `app.py` がなく、起動コマンドを判断できなかった。

**解決策**  
Railway の Settings → Root Directory に `backend` / `frontend` をそれぞれ指定し、サービスを分離した。

---

### エラー 2：uv: command not found

**原因**  
Railway のビルド環境に `uv` がインストールされていない。Nixpacks は Python を検出するが `uv` は標準外ツール。

**解決策**  
`uv` をやめて `pip` + `requirements.txt` に切り替えた。

```bash
cd backend
uv export --format requirements-txt --no-hashes > requirements.txt
git add requirements.txt
git commit -m "add requirements.txt for Railway deploy"
git push
```

---

### エラー 3：'${PORT:-8000}' is not a valid integer

**原因**  
`${PORT:-8000}` というシェルのフォールバック記法が Railway の Start Command フィールドで展開されなかった。

**解決策**  
Python 側でポートを読む方式に変更。

```python
# src/main.py
port = int(os.environ.get("PORT", 8080))
uvicorn.run("src.main:app", host="0.0.0.0", port=port)
```

Start Command をシンプルにした。

```bash
python src/main.py
```

---

### エラー 4：requirements.txt: No such file or directory

**原因**  
`requirements.txt` をローカルで生成したが `git push` を忘れていた。または Root Directory が `backend` なのにルートを参照していた。

**解決策**

```bash
cd backend
uv export --format requirements-txt --no-hashes > requirements.txt
git add . && git commit -m "add requirements.txt" && git push
```

---

### エラー 5：connection to localhost:5432 failed

**原因**  
`config.py` の `postgres_host` のデフォルト値が `localhost` のままで、`DATABASE_URL` 環境変数を渡しても使われなかった。`pydantic-settings` が個別変数で URL を組み立てる設計になっていたため。

```python
# 問題のある設計
DATABASE_URL = (
    f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
)
```

**解決策**  
`database.py` を `os.environ.get()` で直接読む形に修正。

```python
import os

DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
)
```

Railway の Variables に追加。

```bash
DATABASE_URL = postgresql://...(Supabaseの接続文字列)
REDIS_URL    = redis://...(Upstashの接続文字列)
```

---

### エラー 6：Network is unreachable (IPv6)

**原因**  
Supabase の直接接続 (port 5432, `db.xxx.supabase.co`) に Railway から IPv6 で接続しようとしたが到達不能だった。

**解決策**  
Supabase の接続方式を **Session pooler** に変更。

```
# Supabase → Settings → Database → Connection String
Type   : URI
Source : Primary Database
Method : Session pooler  ← ここを変更

# 発行されるURL（IPv4対応）
postgresql://postgres.xxx:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres
```

---

### エラー 7：Uvicorn running on port 8080 → 502

**原因**  
uvicorn が 8080 番で起動したが Railway がそのポートをリッスンしていなかった。`$PORT` の注入タイミングのズレが原因。

**解決策**  
Railway の Variables に `PORT = 8080` を明示的に追加してポートを固定。

---

### エラー 8：$PORT is not valid (Streamlit)

**原因**  
Streamlit が `STREAMLIT_SERVER_PORT` 環境変数を先に読んでしまい `$PORT` が文字列のまま渡った。

**解決策**  
Variables に直接ポート番号を設定。

```bash
STREAMLIT_SERVER_PORT = 8501
STREAMLIT_SERVER_ADDRESS = 0.0.0.0
```

Start Command をシンプルにした。

```bash
streamlit run src/app.py
```

---

### エラー 9：Invalid URL: No scheme supplied

**原因**  
`API_BASE_URL` に `https://` スキームを付け忘れた。

**解決策**

```bash
# 誤
API_BASE_URL = ml-backend-production-217a.up.railway.app

# 正
API_BASE_URL = https://ml-backend-production-217a.up.railway.app
```

---

## 4. 解決の流れ（時系列）

```
Phase 1：ビルドエラーの解消
  ├── No start command → Root Directory を backend に設定
  ├── uv not found → pip + requirements.txt に切り替え
  └── requirements.txt not found → 生成・コミット
      → ビルド成功 ✅

Phase 2：起動エラーの解消
  ├── $PORT 展開問題 → Python 側で os.environ.get('PORT') に変更
  ├── port 8080 ミスマッチ → Variables に PORT=8080 を追加
  └── /docs (Swagger UI) 表示成功 ✅

Phase 3：DB接続の解消
  ├── localhost 接続 → DATABASE_URL がコードに届いていないことが判明
  ├── database.py を os.environ.get('DATABASE_URL') 優先に修正
  ├── Supabase 直接接続 → Session pooler に切り替え
  └── データベース初期化成功 → FastAPI 完全起動 ✅

Phase 4：フロントエンドのデプロイ
  ├── STREAMLIT_SERVER_PORT=8501 を Variables に設定
  ├── API_BASE_URL に https:// を追加
  └── ML Dashboard UI の完全動作を確認 ✅
```

## 5. 最終デプロイ構成

### 5-1. Railway サービス

| サービス | URL |
|---------|-----|
| frontend (Streamlit) | https://ml-frontend-production-e413.up.railway.app |
| backend (FastAPI) | https://ml-backend-production-217a.up.railway.app |

### 5-2. 外部サービス

| サービス | 用途 | リージョン |
|---------|------|-----------|
| Supabase | PostgreSQL | ap-southeast-1 |
| Upstash | Redis | - |

### 5-3. 環境変数（backend）

```bash
DATABASE_URL  = postgresql://postgres.xxx@pooler.supabase.com:5432/postgres
REDIS_URL     = redis://xxx.upstash.io:6379
PORT          = 8080
CORS_ORIGINS  = *
```

### 5-4. 環境変数（frontend）

```bash
API_BASE_URL           = https://ml-backend-production-217a.up.railway.app
STREAMLIT_SERVER_PORT  = 8501
STREAMLIT_SERVER_ADDRESS = 0.0.0.0
```

## 6. 今後の課題

### 短期

- [ ] Supabase Security Advisor の RLS 警告を対応
- [ ] Data Preview の「No preview data available」を修正
- [ ] CORS の `${PUBLIC_HOST}` 未展開を修正
- [ ] README にデプロイ済み URL・スクリーンショット・構成図を追加
- [ ] リポジトリ名のスペルミス修正（ml-dashbord → ml-dashboard）

### 中期（AWS 移行）

- [ ] RDS (PostgreSQL) に切り替え
- [ ] ElastiCache (Redis) に切り替え
- [ ] ECS Fargate で backend / frontend をコンテナ管理
- [ ] ALB でロードバランシング設定
- [ ] Route 53 でカスタムドメイン設定

### ポートフォリオ強化

- [ ] アーキテクチャ図を README に追加
- [ ] CI/CD パイプライン（GitHub Actions）の追加

## 7. 学んだこと

### モノレポ + PaaS のデプロイ

- サービス分割と Root Directory の指定が重要
- 各サービスに独立した設定ファイル（`nixpacks.toml`, `requirements.txt`）が必要

### 環境変数とポート設定

- PaaS の `$PORT` は展開タイミングに注意。Python 側で `os.environ.get()` で読む方が確実
- Streamlit は専用の環境変数（`STREAMLIT_SERVER_PORT`）を優先する

### パッケージ管理

- `uv` など新しいツールは PaaS 環境で未対応の場合がある
- `requirements.txt` へのフォールバックを用意しておく

### データベース設計

- `pydantic-settings` で個別変数から URL を組み立てる設計は、`DATABASE_URL` 一本での切り替えができない
- `os.environ.get()` で上書きできる設計にする

### Supabase 接続

- 直接接続は IPv6 問題が起きやすい
- Session pooler を最初から使うべき

### BaaS の活用

- Supabase / Upstash はポートフォリオ用途でも十分な選択肢
- 無料枠が大きく、技術選定の判断力として評価される
