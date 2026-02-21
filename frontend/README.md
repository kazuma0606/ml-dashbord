# ML Dashboard Frontend

Streamlit-based frontend for the ML Visualization Dashboard.

## Setup

1. Install dependencies using uv:
```bash
cd frontend
uv sync
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env to set API_BASE_URL if needed
```

## Running Locally

```bash
# From the frontend directory
uv run streamlit run app.py
```

The application will be available at http://localhost:8501

## Running with Docker

```bash
# From the project root
docker-compose up frontend
```

## Environment Variables

- `API_BASE_URL`: Backend API URL (default: http://localhost:8000)
- `API_TIMEOUT`: Request timeout in seconds (default: 30)
- `APP_TITLE`: Application title (default: ML Dashboard)
- `APP_ICON`: Application icon emoji (default: 🤖)

## Features

- **Dataset Selection**: Choose from scikit-learn datasets
- **Model Configuration**: Select model type and tune hyperparameters
- **Training**: Train models and view real-time progress
- **Metrics**: View accuracy, F1 score, and other performance metrics
- **Visualizations**: 
  - Data preview with colored labels
  - Confusion matrix heatmap
  - Feature importance charts
  - Experiment history tracking
  - Accuracy trend analysis
- **Model Export**: Download trained models as pickle files
- **Experiment Management**: Save parameters and track experiment history
