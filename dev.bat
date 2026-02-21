@echo off
REM ML Dashboard - Local Development Script (Windows)

echo 🔧 ML Dashboard - Local Development Setup
echo ==========================================

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ uv is not installed. Please install it first:
    echo    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

REM Install dependencies
echo.
echo 📦 Installing dependencies...
echo.

echo Installing backend dependencies...
cd backend
uv sync
cd ..

echo Installing frontend dependencies...
cd frontend
uv sync
cd ..

REM Create .env files if they don't exist
echo.
echo ⚙️  Setting up environment files...

if not exist backend\.env (
    echo Creating backend\.env from backend\.env.example...
    copy backend\.env.example backend\.env
)

if not exist frontend\.env (
    echo Creating frontend\.env from frontend\.env.example...
    copy frontend\.env.example frontend\.env
)

REM Start PostgreSQL and Redis with Docker
echo.
echo 🐳 Starting PostgreSQL and Redis with Docker...
docker-compose up -d postgres redis

echo.
echo ⏳ Waiting for database and cache to be ready...
timeout /t 5 /nobreak >nul

echo.
echo ✅ Development environment ready!
echo.
echo 🚀 To start the services:
echo.
echo    Terminal 1 (Backend):
echo    cd backend
echo    uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo    Terminal 2 (Frontend):
echo    cd frontend
echo    uv run streamlit run src/app.py
echo.
echo 🌐 Access URLs:
echo    Frontend:  http://localhost:8501
echo    Backend:   http://localhost:8000
echo    API Docs:  http://localhost:8000/docs
echo.
pause
