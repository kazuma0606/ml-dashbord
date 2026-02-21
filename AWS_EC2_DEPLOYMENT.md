# AWS EC2 デプロイメントガイド

このガイドでは、AWS EC2インスタンスにML Dashboardをデプロイする手順を説明します。

## 前提条件

- AWSアカウント
- EC2インスタンスへのSSHアクセス
- 基本的なLinuxコマンドの知識

## 推奨EC2インスタンス仕様

- **インスタンスタイプ**: t3.medium 以上（2 vCPU, 4GB RAM）
- **OS**: Ubuntu 22.04 LTS または Amazon Linux 2023
- **ストレージ**: 20GB 以上
- **セキュリティグループ**: 以下のポートを開放
  - SSH (22) - 管理用
  - HTTP (80) - オプション（リバースプロキシ使用時）
  - 8000 - Backend API
  - 8501 - Frontend (Streamlit)

## デプロイ手順

### 1. EC2インスタンスの作成と設定

#### 1.1 EC2インスタンスを起動

1. AWS Management Consoleにログイン
2. EC2ダッシュボードで「インスタンスを起動」をクリック
3. 以下の設定を選択：
   - **名前**: ml-dashboard
   - **AMI**: Ubuntu Server 22.04 LTS
   - **インスタンスタイプ**: t3.medium
   - **キーペア**: 既存のキーペアを選択または新規作成
   - **ストレージ**: 20GB gp3

#### 1.2 セキュリティグループの設定

「ネットワーク設定」で以下のインバウンドルールを追加：

| タイプ | プロトコル | ポート範囲 | ソース | 説明 |
|--------|-----------|-----------|--------|------|
| SSH | TCP | 22 | マイIP | SSH接続用 |
| カスタムTCP | TCP | 8000 | 0.0.0.0/0 | Backend API |
| カスタムTCP | TCP | 8501 | 0.0.0.0/0 | Frontend |

**セキュリティ注意**: 本番環境では、ソースIPを制限することを推奨します。

#### 1.3 Elastic IPの割り当て（推奨）

固定IPアドレスを使用する場合：

1. EC2ダッシュボードで「Elastic IP」を選択
2. 「Elastic IPアドレスを割り当てる」をクリック
3. 割り当てたIPをインスタンスに関連付け

### 2. EC2インスタンスへの接続

```bash
# キーペアのパーミッションを設定
chmod 400 your-key.pem

# EC2インスタンスに接続
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### 3. 必要なソフトウェアのインストール

#### 3.1 システムの更新

```bash
sudo apt update
sudo apt upgrade -y
```

#### 3.2 Dockerのインストール

```bash
# Dockerの公式GPGキーを追加
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Dockerリポジトリを追加
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Dockerをインストール
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Dockerサービスを開始
sudo systemctl start docker
sudo systemctl enable docker

# 現在のユーザーをdockerグループに追加（sudoなしでDockerを使用）
sudo usermod -aG docker $USER

# 変更を適用（再ログインまたは以下のコマンド）
newgrp docker

# Dockerのバージョン確認
docker --version
docker compose version
```

#### 3.3 Gitのインストール

```bash
sudo apt install -y git
```

### 4. アプリケーションのデプロイ

#### 4.1 リポジトリのクローン

```bash
# ホームディレクトリに移動
cd ~

# リポジトリをクローン
git clone <your-repository-url> ml-dashboard
cd ml-dashboard
```

#### 4.2 環境変数の設定

```bash
# .envファイルを作成
cp .env.example .env

# .envファイルを編集
nano .env
```

**重要**: `.env`ファイルで以下を設定：

```bash
# EC2のパブリックIPまたはElastic IPを設定
PUBLIC_HOST=<YOUR-EC2-PUBLIC-IP>

# 本番環境用の強力なパスワードに変更
POSTGRES_PASSWORD=<STRONG-PASSWORD>

# CORS設定を更新
CORS_ORIGINS=http://<YOUR-EC2-PUBLIC-IP>:8501
```

例：
```bash
PUBLIC_HOST=54.123.45.67
POSTGRES_PASSWORD=MySecurePassword123!
CORS_ORIGINS=http://54.123.45.67:8501
```

保存して終了（Ctrl+X, Y, Enter）

#### 4.3 アプリケーションの起動

```bash
# Docker Composeでサービスを起動
docker compose up -d

# ログを確認
docker compose logs -f
```

起動には数分かかる場合があります。以下のメッセージが表示されれば成功です：
- `backend` - "Application startup complete"
- `frontend` - "You can now view your Streamlit app in your browser"

#### 4.4 サービスの状態確認

```bash
# すべてのコンテナが起動しているか確認
docker compose ps

# 特定のサービスのログを確認
docker compose logs backend
docker compose logs frontend
```

### 5. アプリケーションへのアクセス

ブラウザで以下のURLにアクセス：

- **Frontend (メインアプリ)**: `http://<EC2-PUBLIC-IP>:8501`
- **Backend API**: `http://<EC2-PUBLIC-IP>:8000`
- **API ドキュメント**: `http://<EC2-PUBLIC-IP>:8000/docs`

例：
- Frontend: `http://54.123.45.67:8501`
- Backend: `http://54.123.45.67:8000`

### 6. 運用管理

#### 6.1 サービスの管理

```bash
# サービスの停止
docker compose stop

# サービスの再起動
docker compose restart

# サービスの完全停止と削除
docker compose down

# データボリュームも削除（データベースをリセット）
docker compose down -v

# ログの確認
docker compose logs -f

# 特定のサービスのログ
docker compose logs -f backend
```

#### 6.2 アプリケーションの更新

```bash
# 最新のコードを取得
cd ~/ml-dashboard
git pull

# コンテナを再ビルドして起動
docker compose down
docker compose up -d --build

# ログを確認
docker compose logs -f
```

#### 6.3 バックアップ

```bash
# データベースのバックアップ
docker compose exec postgres pg_dump -U postgres ml_dashboard > backup_$(date +%Y%m%d).sql

# バックアップの復元
cat backup_20260221.sql | docker compose exec -T postgres psql -U postgres ml_dashboard
```

#### 6.4 ディスク容量の管理

```bash
# 使用していないDockerリソースをクリーンアップ
docker system prune -a

# ディスク使用量の確認
df -h
docker system df
```

### 7. セキュリティ強化（推奨）

#### 7.1 ファイアウォールの設定

```bash
# UFWをインストール（Ubuntu）
sudo apt install -y ufw

# デフォルトポリシーを設定
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 必要なポートを開放
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # Backend
sudo ufw allow 8501/tcp  # Frontend

# ファイアウォールを有効化
sudo ufw enable

# 状態確認
sudo ufw status
```

#### 7.2 自動起動の設定

システム起動時にDockerコンテナを自動起動：

```bash
# docker-compose.ymlのrestart設定を確認
# すでに "restart: unless-stopped" が設定されているため、
# Dockerサービスが起動すればコンテナも自動起動します

# Dockerサービスの自動起動を確認
sudo systemctl is-enabled docker
```

#### 7.3 Let's Encrypt SSL証明書（オプション）

HTTPSを使用する場合は、Nginxリバースプロキシを設定：

```bash
# Nginxをインストール
sudo apt install -y nginx certbot python3-certbot-nginx

# Nginx設定ファイルを作成
sudo nano /etc/nginx/sites-available/ml-dashboard
```

Nginx設定例：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# 設定を有効化
sudo ln -s /etc/nginx/sites-available/ml-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# SSL証明書を取得
sudo certbot --nginx -d your-domain.com
```

### 8. モニタリング

#### 8.1 リソース使用状況の確認

```bash
# CPU、メモリ使用率
htop

# Dockerコンテナのリソース使用状況
docker stats

# ディスク使用量
df -h
```

#### 8.2 ログの確認

```bash
# リアルタイムログ
docker compose logs -f

# 最新100行のログ
docker compose logs --tail=100

# 特定のサービスのログ
docker compose logs backend --tail=50
```

### 9. トラブルシューティング

#### 問題: コンテナが起動しない

```bash
# ログを確認
docker compose logs

# コンテナの状態を確認
docker compose ps

# 完全にクリーンアップして再起動
docker compose down -v
docker compose up -d
```

#### 問題: フロントエンドがバックエンドに接続できない

1. `.env`ファイルの`CORS_ORIGINS`を確認
2. セキュリティグループでポート8000が開放されているか確認
3. バックエンドが起動しているか確認：`docker compose logs backend`

#### 問題: データベース接続エラー

```bash
# PostgreSQLコンテナが起動しているか確認
docker compose ps postgres

# PostgreSQLのログを確認
docker compose logs postgres

# データベースをリセット
docker compose down -v
docker compose up -d
```

#### 問題: メモリ不足

```bash
# メモリ使用量を確認
free -h

# 不要なコンテナを削除
docker system prune -a

# EC2インスタンスタイプをアップグレード（t3.medium → t3.large）
```

### 10. コスト最適化

#### 10.1 インスタンスの停止

使用しない時間帯はインスタンスを停止：

```bash
# EC2コンソールまたはCLIから停止
aws ec2 stop-instances --instance-ids i-1234567890abcdef0
```

#### 10.2 スポットインスタンスの使用

開発環境では、スポットインスタンスを使用してコストを削減できます。

#### 10.3 リソースの監視

AWS CloudWatchでリソース使用状況を監視し、適切なインスタンスタイプを選択。

## クイックスタートスクリプト

以下のスクリプトを使用すると、セットアップを自動化できます：

```bash
#!/bin/bash
# setup-ec2.sh

set -e

echo "=== ML Dashboard EC2 Setup ==="

# システム更新
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Dockerインストール
echo "Installing Docker..."
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Dockerグループに追加
sudo usermod -aG docker $USER

# Gitインストール
echo "Installing Git..."
sudo apt install -y git

echo "=== Setup Complete ==="
echo "Please log out and log back in for Docker group changes to take effect."
echo "Then clone your repository and run: docker compose up -d"
```

使用方法：
```bash
chmod +x setup-ec2.sh
./setup-ec2.sh
```

## まとめ

これでAWS EC2上でML Dashboardが動作するようになりました！

アクセスURL:
- Frontend: `http://<YOUR-EC2-IP>:8501`
- Backend API: `http://<YOUR-EC2-IP>:8000/docs`

問題が発生した場合は、ログを確認してください：
```bash
docker compose logs -f
```
