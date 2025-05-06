#!/bin/bash

SERVER_NAME="gpu_v100_project51"
LEASE_NAME="gpu_rtx_6000_project51"
FLOATING_IP="192.5.87.67"

echo "Deleting floating IP..."
openstack server remove floating ip "$SERVER_NAME" "$FLOATING_IP" 

echo "Deleting server: $SERVER_NAME"
openstack server delete "$SERVER_NAME"

echo "Deleting lease: $LEASE_NAME"
openstack reservation lease delete "$LEASE_NAME"

echo "Cleanup complete"
