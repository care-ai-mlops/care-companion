#!/bin/bash

VM_ID="$1"
PORT_ID="${VM_ID}_port51"

openstack server delete "${VM_ID}_project51"
openstack port delete "$PORT_ID"
