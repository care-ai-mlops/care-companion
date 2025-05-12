#!/bin/bash

NET_NAME="private_net_project51"
SUBNET_NAME="subnet_project51"
VM_FLAVOR="baremetal"
IMAGE_NAME="CC-Ubuntu24.04-CUDA"
SSH_KEY="key1"
FLOATING_IP="192.5.87.25"
RESERVATION="gpu_compute_gigaio_project51"

echo "Creating lease: $RESERVATION_NAME"
openstack reservation lease create \
  --reservation min=1,max=1,resource_type=physical:host,resource_properties='["=", "$node_type", "compute_gigaio"]' \
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

echo "Done"
