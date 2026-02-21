@echo off
REM ML Dashboard - Startup Script (Windows)

echo 🤖 ML Dashboard - Starting Services
echo ====================================

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from .env.example...
    copy .env.example .env
    echo ✅ .env file created. Please review and adjust settings if needed.
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Start services
echo.
echo 🚀 Starting Docker Compose services...
docker-compose up -d

REM Wait for services to be healthy
echo.
echo ⏳ Waiting for services to be healthy...
timeout /t 5 /nobreak >nul

REM Check service status
echo.
echo 📊 Service Status:
docker-compose ps

echo.
echo ✅ Services started successfully!
echo.
echo 🌐 Access URLs:
echo    Frontend:  http://localhost:8501
echo    Backend:   http://localhost:8000
echo    API Docs:  http://localhost:8000/docs
echo.
echo 📝 Useful commands:
echo    View logs:        docker-compose logs -f
echo    Stop services:    docker-compose down
echo    Restart services: docker-compose restart
echo.
pause
