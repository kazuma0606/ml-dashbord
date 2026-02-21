@echo off
REM ML Dashboard - Stop Script (Windows)

echo 🛑 ML Dashboard - Stopping Services
echo ====================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running.
    exit /b 1
)

REM Stop services
echo.
echo 🛑 Stopping Docker Compose services...
docker-compose down

echo.
echo ✅ Services stopped successfully!
echo.
echo 💡 To remove all data (database and cache), run:
echo    docker-compose down -v
echo.
pause
