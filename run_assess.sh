#!/bin/bash

# Define GB frequencies and corresponding GPU IDs
declare -a gb_frequencies=("2e-4" "3e-4" "2e-3" "3e-3")
declare -a gpus=("0" "1" "4" "7")

# Loop through frequencies and GPUs
for i in "${!gb_frequencies[@]}"; do
    freq=${gb_frequencies[$i]}
    gpu=${gpus[$i]}
    
    echo "Running with GB frequency $freq on GPU $gpu"
    nohup python assess_impact.py "$freq" "$gpu" > "out_$freq.out" &
done

echo "All processes started."