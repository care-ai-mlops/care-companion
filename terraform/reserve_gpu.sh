#!/bin/bash

NET_NAME="private_net_project51"
SUBNET_NAME="subnet_project51"
VM_FLAVOR="baremetal"
IMAGE_NAME="CC-Ubuntu24.04"
SSH_KEY="key1"
FLOATING_IP="192.5.87.25"
RESERVATION="gpu_compute_gigaio_project51"

echo "Creating lease: $RESERVATION_NAME"
openstack reservation lease create \
  --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_rtx_6000"]' \
  "$RESERVATION"

echo "Waiting for lease to become ACTIVE..."
until openstack reservation lease show "$RESERVATION" | grep -i '| status\s*| ACTIVE' > /dev/null; do
    openstack reservation lease show "$RESERVATION" | grep -i '| status\s*|'
    sleep 2
done

RESERVATION_ID=$(openstack reservation lease show "$RESERVATION" | grep '"id":' | awk -F'"' '{print $4}' | tail -1)
echo "Reservation ID: $RESERVATION_ID"


echo "Done"
