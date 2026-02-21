#!/bin/bash
# ML Dashboard Deployment Script
# Quick deployment and management script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${NC}ℹ $1${NC}"
}

# Check if .env exists
check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration:"
        print_info "  nano .env"
        print_info "Set PUBLIC_HOST to your EC2 public IP or domain"
        exit 1
    fi
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running or not installed"
        print_info "Please install Docker first: ./setup-ec2.sh"
        exit 1
    fi
}

# Start services
start() {
    print_info "Starting ML Dashboard..."
    check_env
    check_docker
    
    docker compose up -d
    
    print_success "Services started!"
    print_info "Waiting for services to be ready..."
    sleep 5
    
    status
}

# Stop services
stop() {
    print_info "Stopping ML Dashboard..."
    docker compose stop
    print_success "Services stopped!"
}

# Restart services
restart() {
    print_info "Restarting ML Dashboard..."
    docker compose restart
    print_success "Services restarted!"
    sleep 3
    status
}

# Show status
status() {
    print_info "Service Status:"
    docker compose ps
    
    echo ""
    print_info "Access URLs:"
    
    # Try to get PUBLIC_HOST from .env
    if [ -f .env ]; then
        PUBLIC_HOST=$(grep PUBLIC_HOST .env | cut -d '=' -f2)
        if [ -n "$PUBLIC_HOST" ] && [ "$PUBLIC_HOST" != "localhost" ]; then
            echo "  Frontend: http://${PUBLIC_HOST}:8501"
            echo "  Backend:  http://${PUBLIC_HOST}:8000"
            echo "  API Docs: http://${PUBLIC_HOST}:8000/docs"
        else
            echo "  Frontend: http://localhost:8501"
            echo "  Backend:  http://localhost:8000"
            echo "  API Docs: http://localhost:8000/docs"
        fi
    fi
}

# Show logs
logs() {
    if [ -z "$1" ]; then
        docker compose logs -f
    else
        docker compose logs -f "$1"
    fi
}

# Update application
update() {
    print_info "Updating ML Dashboard..."
    
    # Pull latest code
    print_info "Pulling latest code..."
    git pull
    
    # Rebuild and restart
    print_info "Rebuilding containers..."
    docker compose down
    docker compose up -d --build
    
    print_success "Update complete!"
    sleep 3
    status
}

# Clean up
clean() {
    print_warning "This will remove all containers and volumes (including database data)!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_info "Cleaning up..."
        docker compose down -v
        print_success "Cleanup complete!"
    else
        print_info "Cleanup cancelled"
    fi
}

# Backup database
backup() {
    print_info "Creating database backup..."
    
    BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
    docker compose exec -T postgres pg_dump -U postgres ml_dashboard > "$BACKUP_FILE"
    
    print_success "Backup created: $BACKUP_FILE"
}

# Show help
help() {
    echo "ML Dashboard Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show service status and URLs"
    echo "  logs      - Show logs (optional: specify service name)"
    echo "  update    - Pull latest code and rebuild"
    echo "  backup    - Backup database"
    echo "  clean     - Remove all containers and volumes"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh start"
    echo "  ./deploy.sh logs backend"
    echo "  ./deploy.sh update"
}

# Main script
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    update)
        update
        ;;
    backup)
        backup
        ;;
    clean)
        clean
        ;;
    help|--help|-h|"")
        help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        help
        exit 1
        ;;
esac
