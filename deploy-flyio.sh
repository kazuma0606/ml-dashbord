#!/bin/bash
# Fly.io Deployment Script for ML Dashboard

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if flyctl is installed
check_flyctl() {
    if ! command -v flyctl &> /dev/null; then
        print_error "flyctl is not installed"
        print_info "Install it from: https://fly.io/docs/hands-on/install-flyctl/"
        exit 1
    fi
}

# Check if logged in
check_auth() {
    if ! flyctl auth whoami &> /dev/null; then
        print_error "Not logged in to Fly.io"
        print_info "Run: flyctl auth login"
        exit 1
    fi
}

# Deploy backend
deploy_backend() {
    print_header "Deploying Backend"
    
    cd backend
    
    if [ ! -f "fly.toml" ]; then
        print_error "fly.toml not found in backend/"
        exit 1
    fi
    
    print_info "Deploying backend..."
    flyctl deploy
    
    print_success "Backend deployed!"
    
    # Get backend URL
    BACKEND_URL=$(flyctl info --json | grep -o '"Hostname":"[^"]*"' | cut -d'"' -f4)
    print_info "Backend URL: https://${BACKEND_URL}"
    
    cd ..
}

# Deploy frontend
deploy_frontend() {
    print_header "Deploying Frontend"
    
    cd frontend
    
    if [ ! -f "fly.toml" ]; then
        print_error "fly.toml not found in frontend/"
        exit 1
    fi
    
    print_info "Deploying frontend..."
    flyctl deploy
    
    print_success "Frontend deployed!"
    
    # Get frontend URL
    FRONTEND_URL=$(flyctl info --json | grep -o '"Hostname":"[^"]*"' | cut -d'"' -f4)
    print_info "Frontend URL: https://${FRONTEND_URL}"
    
    cd ..
}

# Setup databases
setup_databases() {
    print_header "Setting up Databases"
    
    print_info "Creating PostgreSQL database..."
    flyctl postgres create --name ml-dashboard-db --region nrt || print_warning "Database may already exist"
    
    print_info "Creating Redis cache..."
    flyctl redis create --name ml-dashboard-redis --region nrt || print_warning "Redis may already exist"
    
    print_success "Databases setup complete!"
}

# Configure backend secrets
configure_backend() {
    print_header "Configuring Backend"
    
    cd backend
    
    print_info "Setting backend environment variables..."
    
    read -p "Enter PostgreSQL password: " POSTGRES_PASSWORD
    
    flyctl secrets set \
        POSTGRES_HOST=ml-dashboard-db.internal \
        POSTGRES_PORT=5432 \
        POSTGRES_USER=postgres \
        POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        POSTGRES_DB=ml_dashboard \
        REDIS_HOST=ml-dashboard-redis.internal \
        REDIS_PORT=6379 \
        REDIS_DB=0 \
        CORS_ORIGINS="*"
    
    print_success "Backend configured!"
    
    cd ..
}

# Configure frontend secrets
configure_frontend() {
    print_header "Configuring Frontend"
    
    cd frontend
    
    print_info "Getting backend URL..."
    cd ../backend
    BACKEND_URL=$(flyctl info --json | grep -o '"Hostname":"[^"]*"' | cut -d'"' -f4)
    cd ../frontend
    
    print_info "Setting frontend environment variables..."
    flyctl secrets set API_BASE_URL="https://${BACKEND_URL}"
    
    print_success "Frontend configured!"
    
    cd ..
}

# Show status
show_status() {
    print_header "Application Status"
    
    print_info "Backend status:"
    cd backend
    flyctl status
    BACKEND_URL=$(flyctl info --json | grep -o '"Hostname":"[^"]*"' | cut -d'"' -f4)
    cd ..
    
    echo ""
    print_info "Frontend status:"
    cd frontend
    flyctl status
    FRONTEND_URL=$(flyctl info --json | grep -o '"Hostname":"[^"]*"' | cut -d'"' -f4)
    cd ..
    
    echo ""
    print_header "Access URLs"
    echo "Frontend: https://${FRONTEND_URL}"
    echo "Backend:  https://${BACKEND_URL}"
    echo "API Docs: https://${BACKEND_URL}/docs"
}

# Show logs
show_logs() {
    SERVICE=$1
    
    if [ "$SERVICE" = "backend" ]; then
        cd backend
        flyctl logs -f
    elif [ "$SERVICE" = "frontend" ]; then
        cd frontend
        flyctl logs -f
    else
        print_error "Invalid service. Use 'backend' or 'frontend'"
        exit 1
    fi
}

# Full deployment
full_deploy() {
    print_header "Full Deployment to Fly.io"
    
    check_flyctl
    check_auth
    
    print_warning "This will deploy the entire application to Fly.io"
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_info "Deployment cancelled"
        exit 0
    fi
    
    setup_databases
    configure_backend
    deploy_backend
    configure_frontend
    deploy_frontend
    
    print_success "Deployment complete!"
    show_status
}

# Help
show_help() {
    echo "Fly.io Deployment Script for ML Dashboard"
    echo ""
    echo "Usage: ./deploy-flyio.sh [command]"
    echo ""
    echo "Commands:"
    echo "  full              - Full deployment (databases + backend + frontend)"
    echo "  setup-db          - Setup PostgreSQL and Redis"
    echo "  deploy-backend    - Deploy backend only"
    echo "  deploy-frontend   - Deploy frontend only"
    echo "  configure-backend - Configure backend environment variables"
    echo "  configure-frontend- Configure frontend environment variables"
    echo "  status            - Show application status"
    echo "  logs <service>    - Show logs (backend or frontend)"
    echo "  help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy-flyio.sh full"
    echo "  ./deploy-flyio.sh deploy-backend"
    echo "  ./deploy-flyio.sh logs backend"
}

# Main
case "$1" in
    full)
        full_deploy
        ;;
    setup-db)
        check_flyctl
        check_auth
        setup_databases
        ;;
    deploy-backend)
        check_flyctl
        check_auth
        deploy_backend
        ;;
    deploy-frontend)
        check_flyctl
        check_auth
        deploy_frontend
        ;;
    configure-backend)
        check_flyctl
        check_auth
        configure_backend
        ;;
    configure-frontend)
        check_flyctl
        check_auth
        configure_frontend
        ;;
    status)
        check_flyctl
        check_auth
        show_status
        ;;
    logs)
        check_flyctl
        check_auth
        show_logs "$2"
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
