#!/bin/bash
# ML Dashboard EC2 Setup Script
# This script automates the setup of Docker and dependencies on Ubuntu EC2

set -e

echo "=========================================="
echo "ML Dashboard EC2 Setup Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "Please do not run this script as root"
    exit 1
fi

# Update system
echo "[1/5] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Docker
echo "[2/5] Installing Docker..."
sudo apt install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
fi

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker packages
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
echo "[3/5] Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
echo "[4/5] Adding user to docker group..."
sudo usermod -aG docker $USER

# Install Git
echo "[5/5] Installing Git..."
sudo apt install -y git

# Install additional useful tools
echo "Installing additional tools..."
sudo apt install -y htop curl wget nano

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in (or run: newgrp docker)"
echo "2. Clone your repository:"
echo "   git clone <your-repo-url> ml-dashboard"
echo "3. Navigate to the directory:"
echo "   cd ml-dashboard"
echo "4. Copy and edit the environment file:"
echo "   cp .env.example .env"
echo "   nano .env"
echo "5. Start the application:"
echo "   docker compose up -d"
echo ""
echo "Verify Docker installation:"
docker --version
docker compose version
echo ""
