#!/bin/bash
set -e

echo "Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh

echo "Adding current user to docker group..."
sudo groupadd -f docker
sudo usermod -aG docker $USER
sudo chmod 666 /var/run/docker.sock

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
