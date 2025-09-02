#!/bin/bash

# Spine Analyzer Quick Start Script
# This script helps you quickly set up and run the Spine Analyzer system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_info "Docker is installed ✓"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_info "Docker Compose is installed ✓"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_info ".env file created. Please update it with your configuration."
            print_info "Opening .env file for editing..."
            ${EDITOR:-nano} .env
        else
            print_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    else
        print_info ".env file exists ✓"
    fi
}

# Check if NVIDIA GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "NVIDIA GPU detected ✓"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "NVIDIA GPU not detected. Triton may run in CPU mode (slower)."
        read -p "Continue without GPU? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Pull Docker images
pull_images() {
    print_info "Pulling Docker images..."
    docker-compose pull
}

# Build custom images
build_images() {
    print_info "Building custom Docker images..."
    docker-compose build
}

# Start services
start_services() {
    print_info "Starting services..."
    docker-compose up -d
    
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    print_info "Checking service status..."
    docker-compose ps
}

# Test the installation
test_installation() {
    print_info "Testing the installation..."
    
    # Check if Orthanc is responding
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8042/system | grep -q "200"; then
        print_info "Orthanc is running ✓"
    else
        print_warning "Orthanc is not responding. Check the logs with: docker-compose logs orthanc"
    fi
    
    # Check if Pipeline API is responding
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs | grep -q "200"; then
        print_info "Pipeline API is running ✓"
        print_info "API documentation available at: http://localhost:8000/docs"
    else
        print_warning "Pipeline API is not responding. Check the logs with: docker-compose logs pipeline"
    fi
    
    # Check if Triton is responding
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready | grep -q "200"; then
        print_info "Triton Inference Server is running ✓"
    else
        print_warning "Triton is not responding. Check the logs with: docker-compose logs triton"
    fi
}

# Main menu
show_menu() {
    echo
    echo "================================"
    echo "   Spine Analyzer Quick Start   "
    echo "================================"
    echo "1. Full Setup (Recommended for first time)"
    echo "2. Start Services"
    echo "3. Stop Services"
    echo "4. View Logs"
    echo "5. Test Installation"
    echo "6. Process Sample Study"
    echo "7. Clean Up (Remove all data)"
    echo "8. Exit"
    echo "================================"
    read -p "Select an option [1-8]: " choice
    
    case $choice in
        1)
            print_info "Starting full setup..."
            check_docker
            check_docker_compose
            check_env_file
            check_gpu
            pull_images
            build_images
            start_services
            test_installation
            print_info "Setup complete! 🎉"
            print_info "Access points:"
            print_info "  - Orthanc Web UI: http://localhost:8042"
            print_info "  - Pipeline API: http://localhost:8000/docs"
            ;;
        2)
            print_info "Starting services..."
            docker-compose up -d
            print_info "Services started."
            ;;
        3)
            print_info "Stopping services..."
            docker-compose down
            print_info "Services stopped."
            ;;
        4)
            print_info "Showing logs (press Ctrl+C to exit)..."
            docker-compose logs -f
            ;;
        5)
            test_installation
            ;;
        6)
            print_info "Processing sample study..."
            read -p "Enter Orthanc Study ID: " study_id
            curl -X POST http://localhost:8000/process-study/ \
                -H "Content-Type: application/x-www-form-urlencoded" \
                -d "study_id=$study_id"
            ;;
        7)
            print_warning "This will remove all containers, volumes, and data!"
            read -p "Are you sure? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose down -v
                print_info "Clean up complete."
            fi
            ;;
        8)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option. Please select 1-8."
            ;;
    esac
}

# Main loop
while true; do
    show_menu
done