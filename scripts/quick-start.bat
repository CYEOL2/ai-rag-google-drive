@echo off
chcp 65001 >nul 2>&1

echo.
echo ========================================
echo  RAG App Quick Start (using compose)
echo ========================================
echo.

cd /d "%~dp0.."

echo Current running containers:
docker ps --format "table {{.Names}}\t{{.Status}}"

echo.
echo Starting all services with docker-compose...
echo (Existing containers will be kept running)

docker-compose -f docker-compose.full.yml up -d

if errorlevel 1 (
    echo ERROR: Failed to start services
    echo.
    echo Troubleshooting:
    echo   1. Check if Dockerfile exists: dir Dockerfile
    echo   2. Try manual build: docker build -t ai-project_rag-app .
    echo   3. Check logs: docker-compose -f docker-compose.full.yml logs
    pause
    exit /b 1
)

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul 2>&1

echo.
echo Final container status:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo ========================================
echo    SUCCESS: All services started!
echo ========================================
echo.
echo Web interface: http://localhost:8000
echo Container status: docker ps
echo View logs: docker-compose -f docker-compose.full.yml logs -f rag-app
echo Stop all: docker-compose -f docker-compose.full.yml down
echo.
echo Only the RAG app was built/started. Ollama and Qdrant were kept running.
echo.

set /p open_browser="Open web browser? (y/N): "
if /i "%open_browser%"=="y" (
    start http://localhost:8000
)

pause
