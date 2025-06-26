#!/bin/bash

echo ""
echo "========================================"
echo " RAG App Quick Start (using compose)"
echo "========================================"
echo ""

# Change to project root directory
cd "$(dirname "$0")/.."

echo "Current running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "Starting all services with docker-compose..."
echo "(Existing containers will be kept running)"

docker-compose -f docker-compose.full.yml up -d

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to start services"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if Dockerfile exists: ls -la Dockerfile"
    echo "  2. Try manual build: docker build -t ai-project_rag-app ."
    echo "  3. Check logs: docker-compose -f docker-compose.full.yml logs"
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "Final container status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "========================================"
echo "    SUCCESS: All services started!"
echo "========================================"
echo ""
echo "Web interface: http://localhost:8000"
echo "Container status: docker ps"
echo "View logs: docker-compose -f docker-compose.full.yml logs -f rag-app"
echo "Stop all: docker-compose -f docker-compose.full.yml down"
echo ""
echo "Only the RAG app was built/started. Ollama and Qdrant were kept running."
echo ""

read -p "Open web browser? (y/N): " open_browser
if [[ $open_browser =~ ^[Yy]$ ]]; then
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:8000
    elif command -v open > /dev/null; then
        open http://localhost:8000
    else
        echo "Please manually open http://localhost:8000"
    fi
fi

echo ""
echo "Press Enter to exit..."
read
