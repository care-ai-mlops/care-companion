#!/bin/bash

NET_NAME="private_net_project51"
SUBNET_NAME="subnet_project51"
VM_FLAVOR="baremetal"
IMAGE_NAME="CC-Ubuntu24.04-CUDA"
SSH_KEY="key1"
FLOATING_IP="192.5.87.67"
RESERVATION="gpu_rtx_6000_project51"

echo "Creating lease: $RESERVATION_NAME"
openstack reservation lease create \
  --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "gpu_rtx_6000"]' \
  "$RESERVATION"

LEASE_ID=$(openstack reservation lease show "$RESERVATION" -f value -c id)

echo "Waiting for lease to become ACTIVE..."
while true; do
  STATUS=$(openstack reservation lease show "$LEASE_ID" -f value -c status)
  if [[ "$STATUS" == "ACTIVE" ]]; then
    echo "Lease is ACTIVE."
    break
  elif [[ "$STATUS" == "ERROR" ]]; then
    echo "Lease failed."
    exit 1
  fi
  sleep 2
done

RESERVE_ID=$(openstack reservation lease show "$RESERVATION" | grep '"id":' | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | tail -1)
echo "Reservation ID: $RESERVE_ID"

echo "Creating server: $INSTANCE_NAME"
openstack server create \
  --image "$IMAGE_NAME" \
  --flavor "$VM_FLAVOR" \
  --key-name "$SSH_KEY" \
  --network sharednet1 \
  --security-group default \
  --security-group allow-ssh \
  --hint reservation="$RESERVE_ID" \
  "gpu_v100_project51"

echo "Waiting for server to become ACTIVE..."
sleep 600

echo "Assigning floating IP $FLOATING_IP to gpu_v100_project51"
openstack server add floating ip "gpu_v100_project51" "$FLOATING_IP"

echo "Done"