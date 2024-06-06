#!/bin/zsh

# Create a new sweep and capture the output
output=$(wandb sweep sweep-walle.yaml 2>&1)

# Extract the sweep ID using grep and awk
sweep_id=$(echo "$output" | grep "wandb: Creating sweep with ID:" | awk '{print $NF}')
echo "Sweep ID: $sweep_id"

# Check if the sweep_id was captured
if [[ -z "$sweep_id" ]]; then
    echo "Failed to create sweep or capture sweep ID."
    exit 1
else
    echo "Sweep ID captured: $sweep_id"
    # Start the wandb agent with the captured sweep ID
    wandb agent $sweep_id
fi
