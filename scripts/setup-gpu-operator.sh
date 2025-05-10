#!/bin/bash

# Create namespace and set security context
kubectl create ns gpu-operator
kubectl label --overwrite ns gpu-operator pod-security.kubernetes.io/enforce=privileged

# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU operator
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v25.3.0

# Wait for operator to be ready
echo "Waiting for GPU operator to be ready..."
sleep 60

# Apply time-slicing configuration
kubectl apply -f k8s/platform/templates/gpu-operator.yaml

# Configure time-slicing
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all", "default": "any"}}}}'

# Wait for configuration to be applied
echo "Waiting for time-slicing configuration to be applied..."
sleep 60

# Verify GPU time-slicing
echo "Verifying GPU time-slicing..."
kubectl apply -f k8s/platform/templates/gpu-operator.yaml
kubectl get pods -n gpu-operator
kubectl logs -n gpu-operator deploy/time-slicing-verification

# Clean up verification deployment
kubectl delete -f k8s/platform/templates/gpu-operator.yaml 