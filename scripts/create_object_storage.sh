#!bin/bash

set -e
echo "Installing rclone..."

echo "Mounting object storage..."
curl https://rclone.org/install.sh | sudo bash

echo "Configuring rclone..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
mkdir -p ~/.config/rclone
nano  ~/.config/rclone/rclone.conf

echo "Checking for Object Storage..."
rclone lsd chi_tacc:object-persist-project51
if [ $? -ne 0 ]; then
    echo "Object storage not found. Please check your rclone configuration."
    exit 1
fi


if [ ! -d "/mnt/object/" ]; then
    echo "Creating mount point..."
    sudo mkdir -p /mnt/object
fi

echo "Setting permissions for mount point..."
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object


echo "Mounting object storage..."
rclone mount chi_tacc:object-persist-project51 /mnt/object --allow-other --vfs-cache-mode writes --daemon

echo "Object storage mounted at /mnt/object"
ls -l /mnt/object
echo "Mounting complete. You can now access your object storage at /mnt/object."