#!/bin/bash
set -e

echo "Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh

echo "Adding current user to docker group..."
sudo groupadd -f docker
sudo usermod -aG docker $USER
sudo chmod 666 /var/run/docker.sock

echo "Setting up NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "Updating package lists..."
sudo apt update

echo "Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

echo "Configuring Docker runtime for NVIDIA..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "Updating Docker daemon configuration for cgroups..."
# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    sudo apt-get install -y jq
fi

# Update daemon.json with cgroup driver setting
sudo jq 'if has("exec-opts") then . else . + {"exec-opts": ["native.cgroupdriver=cgroupfs"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json

echo "Restarting Docker service..."
sudo systemctl restart docker

# Install nvtop
sudo apt update
sudo apt -y install nvtop

# Install rclone 
curl https://rclone.org/install.sh | sudo bash

# Update rclone config file
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
mkdir -p ~/.config/rclone
