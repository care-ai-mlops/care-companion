# OpenStack Resource Management for KVM@TACC

## Prerequisites
- Ensure you have OpenStack CLI installed and your `openrc` file sourced.

## Commands

1. **Make scripts executable**:
    ```bash
    chmod +x openstack/create_resources.sh
    chmod +x openstack/delete_resources.sh
    ```

2. **Create resources (instance and port)**:
    ```bash
    ./openstack/create_resources.sh instance_name 192.112.0.10
    ```

3. **Assign a floating IP to the instance**:
    ```bash
    openstack server add floating ip instance_name floating_ip
    ```
    
4. **Delete resources (instance and port)**:
    ```bash
    ./openstack/delete_resources.sh node1
    ```

