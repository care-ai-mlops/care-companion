#!/bin/bash
set -e

echo "Setting up Block Storage..."
lsblk

echo "Checking for Block Storage..."
lsblk | grep vdb
if [ $? -ne 0 ]; then
    echo "Block storage not found. Please check your configuration."
    exit 1
fi

echo "Creating partition..."
sudo parted /dev/vdb mklabel gpt
sudo parted /dev/vdb mkpart primary ext4 0% 100%

echo "Formatting partition..."
sudo mkfs.ext4 /dev/vdb1

echo "Creating mount point..."
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block

echo "Setting permissions for mount point..."
sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block

echo "Checking for the mount..."
df -h | grep /mnt/block