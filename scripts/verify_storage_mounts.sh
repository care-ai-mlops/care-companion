#!/bin/bash

# Function to check if a directory exists and is mounted
check_mount() {
    local mount_point=$1
    local storage_type=$2
    
    echo "Checking $storage_type mount at $mount_point..."
    
    if [ ! -d "$mount_point" ]; then
        echo "Error: Directory $mount_point does not exist"
        return 1
    fi
    
    if ! mountpoint -q "$mount_point"; then
        echo "Error: $mount_point is not a mount point"
        return 1
    fi
    
    echo "$storage_type is correctly mounted at $mount_point"
    return 0
}

# Function to check block storage subdirectories
check_block_storage_dirs() {
    local base_dir="/mnt/block"
    local dirs=("ray" "labelstudio" "prometheus" "grafana")
    
    echo "Checking block storage subdirectories..."
    
    for dir in "${dirs[@]}"; do
        local full_path="$base_dir/$dir"
        if [ ! -d "$full_path" ]; then
            echo "Error: Block storage directory $full_path does not exist"
            return 1
        fi
        echo "Block storage directory $full_path exists"
    done
    
    return 0
}

# Function to check object storage access
check_object_storage() {
    local mount_point="/mnt/object"
    
    echo "Checking object storage access..."
    
    if [ ! -d "$mount_point" ]; then
        echo "Error: Object storage directory $mount_point does not exist"
        return 1
    fi
    
    # Try to list contents of object storage
    if ! ls "$mount_point" > /dev/null 2>&1; then
        echo "Error: Cannot access object storage at $mount_point"
        return 1
    fi
    
    echo "Object storage is accessible at $mount_point"
    return 0
}

# Main verification
echo "Starting storage mount verification..."

# Check block storage
check_mount "/mnt/block" "Block storage"
block_status=$?

# Check block storage subdirectories
check_block_storage_dirs
block_dirs_status=$?

# Check object storage
check_mount "/mnt/object" "Object storage"
object_status=$?

# Check object storage access
check_object_storage
object_access_status=$?

# Print summary
echo -e "\nVerification Summary:"
echo "Block Storage Mount: $([ $block_status -eq 0 ] && echo "✅" || echo "❌")"
echo "Block Storage Directories: $([ $block_dirs_status -eq 0 ] && echo "✅" || echo "❌")"
echo "Object Storage Mount: $([ $object_status -eq 0 ] && echo "✅" || echo "❌")"
echo "Object Storage Access: $([ $object_access_status -eq 0 ] && echo "✅" || echo "❌")"

# Exit with error if any check failed
if [ $block_status -ne 0 ] || [ $block_dirs_status -ne 0 ] || [ $object_status -ne 0 ] || [ $object_access_status -ne 0 ]; then
    exit 1
fi

exit 0 