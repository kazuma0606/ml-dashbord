#!/bin/bash
# ML Dashboard - Stop Script (Unix/Linux/macOS)

set -e

echo "🛑 ML Dashboard - Stopping Services"
echo "===================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running."
    exit 1
fi

# Stop services
echo ""
echo "🛑 Stopping Docker Compose services..."
docker-compose down

echo ""
echo "✅ Services stopped successfully!"
echo ""
echo "💡 To remove all data (database and cache), run:"
echo "   docker-compose down -v"
echo ""
