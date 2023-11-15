#!/bin/bash

# Define an indexed array with experiment names followed by their parameters
experiments=(
    "20231115_raw_lr1x10_3 --batch-size=16 --gpu-ram=16384 --learning-rate=1e-3 --model-variant=u_net"
    "20231115_gn_lr1x10_3 --batch-size=16 --gpu-ram=16384 --learning-rate=1e-3 --model-variant=u_net_gn"
    "20231115_res_lr1x10_3 --batch-size=16 --gpu-ram=16384 --learning-rate=1e-3 --model-variant=u_net_res"
    "20231115_raw_lr1x10_5 --batch-size=16 --gpu-ram=16384 --learning-rate=1e-5 --model-variant=u_net"
    "20231115_gn_lr1x10_5 --batch-size=16 --gpu-ram=16384 --learning-rate=1e-5 --model-variant=u_net_gn"
    "20231115_res_lr1x10_5 --batch-size=16 --gpu-ram=16384 --learning-rate=1e-5 --model-variant=u_net_res"
)

# Function to execute the Python script with given parameters
execute_experiment() {
    # Split the input into an array based on space
    IFS=' ' read -ra parts <<< "$1"

    # Extract experiment name and parameters
    exp_name="${parts[0]}"

    # Create experiment folder
    mkdir -p ./experiments/$exp_name

    # Remove the first element (experiment name)
    unset parts[0]

    # Join the remaining parts (parameters) into a single string
    params="${parts[*]}"

    echo "Starting $exp_name with parameters: $params"
    python train.py $params > "./experiments/$exp_name/output_${exp_name}.log" 2>&1 &
    pid=$!

    echo "Experiment $exp_name started with PID: $pid"

    # Wait for the experiment to finish
    wait $pid
    echo "Experiment $exp_name completed."

    mv ./unet_denoise.h5 ./experiments/$exp_name/
    mv ./logs ./experiments/$exp_name/
}

# Loop through each experiment entry and execute them
for exp_entry in "${experiments[@]}"; do
    execute_experiment "$exp_entry"
done

echo "All experiments have been executed."
