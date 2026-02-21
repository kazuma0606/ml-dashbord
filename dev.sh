#!/bin/bash
# ML Dashboard - Local Development Script (Unix/Linux/macOS)

set -e

echo "🔧 ML Dashboard - Local Development Setup"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo ""

echo "Installing backend dependencies..."
cd backend
uv sync
cd ..

echo "Installing frontend dependencies..."
cd frontend
uv sync
cd ..

# Create .env files if they don't exist
echo ""
echo "⚙️  Setting up environment files..."

if [ ! -f backend/.env ]; then
    echo "Creating backend/.env from backend/.env.example..."
    cp backend/.env.example backend/.env
fi

if [ ! -f frontend/.env ]; then
    echo "Creating frontend/.env from frontend/.env.example..."
    cp frontend/.env.example frontend/.env
fi

# Start PostgreSQL and Redis with Docker
echo ""
echo "🐳 Starting PostgreSQL and Redis with Docker..."
docker-compose up -d postgres redis

echo ""
echo "⏳ Waiting for database and cache to be ready..."
sleep 5

echo ""
echo "✅ Development environment ready!"
echo ""
echo "🚀 To start the services:"
echo ""
echo "   Terminal 1 (Backend):"
echo "   cd backend"
echo "   uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "   Terminal 2 (Frontend):"
echo "   cd frontend"
echo "   uv run streamlit run src/app.py"
echo ""
echo "🌐 Access URLs:"
echo "   Frontend:  http://localhost:8501"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
