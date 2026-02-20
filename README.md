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

## 開発状況

現在、仕様書作成フェーズが完了しました。実装はこれから開始します。

詳細な仕様は `.kiro/specs/ml-visualization-dashboard/` を参照してください：
- `requirements.md` - 要件定義書
- `design.md` - 設計書
- `tasks.md` - 実装計画

## ライセンス

MIT
