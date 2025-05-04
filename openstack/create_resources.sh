#!/bin/bash

NET_NAME="private_net_project51"
SUBNET_NAME="subnet_project51"
VM_FLAVOR="m1.medium"
IMAGE_NAME="CC-Ubuntu24.04"
SSH_KEY="key1"
FLOATING_IP="129.114.27.163" 

VM_ID="$1"
STATIC_IP="$2"

PORT_ID="${VM_ID}_port51"

openstack port create \
  --network "$NET_NAME" \
  --fixed-ip subnet="$SUBNET_NAME",ip-address="$STATIC_IP" \
  --disable-port-security \
  "$PORT_ID"

openstack server create \
  --image "$IMAGE_NAME" \
  --flavor "$VM_FLAVOR" \
  --port "$PORT_ID" \
  --network sharednet1 \
  --key-name "$SSH_KEY" \
  --security-group default \
  --security-group allow-ssh \
  --security-group allow-http-80 \
  "${VM_ID}_project51"
