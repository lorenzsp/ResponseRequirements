# run with: nohup bash run_assess.sh > out.log &

# Define GB frequencies and corresponding GPU IDs
declare -a gb_frequencies=("1e-4" "2e-4" "5e-4" "1e-3" "2e-3" "5e-3" "1e-2")
declare -a gpus=("1" "1" "1" "1" "1" "1" "1")

# Loop through frequencies and GPUs
for i in "${!gb_frequencies[@]}"; do
    freq=${gb_frequencies[$i]}
    gpu=${gpus[$i]}
    
    echo "Running with GB frequency $freq on GPU $gpu"
    # nohup python assess_impact.py "$freq" "$gpu" > "out_$freq.out" &
    python assess_impact.py "$freq" "$gpu"
done

echo "All processes started."