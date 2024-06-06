#!/bin/zsh

# Aim: Sweep linear model (i.e. without the graph conv layers) on AsEPv1.1.0

set -e

# Set configuration variables
BASE=$(dirname $(realpath $0))
WD=$(dirname $BASE)

# Create a new sweep and capture the output
output=$(wandb sweep sweep-walle.yaml 2>&1)

# Extract the sweep ID using grep and awk
sweepId=$(echo "$output" | grep "wandb: Creating sweep with ID:" | awk '{print $NF}')
echo "Sweep ID: $sweepId"

# NOTE: change the CUDA device to the one you want to use
cudaDeviceIndex=0

# start agent
CUDA_VISIBLE_DEVICES=$cudaDeviceIndex wandb agent $sweepId