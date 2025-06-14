#!/bin/bash

model_path="GSAI-ML/LLaDA-8B-Instruct"
checkpoint_path="/cephfs/xukangping/code/d1/SFT/outputs/llada-s1/checkpoint-320"
name="llada-s1-sft"

# Take out the basename as the output dir
# If has name, use the name, or else use the model basename
if [ -n "$name" ]; then
  output_dir=./eval_results/$name
else
  output_dir=./eval_results/$(basename $model_path)
fi

# Configuration variables
GPU_IDS=(0 1 2 3 4 5 6 7)

MASTER_PORT=29411

# Arrays of tasks and generation lengths
TASKS=("countdown" "sudoku" "math" "gsm8k")
GEN_LENGTHS=(128 256)

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    command="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir $output_dir \
      --model_path $model_path
    "

    # If given checkpoint_path, add it to the command
    if [ -n "$checkpoint_path" ]; then
      command="$command --checkpoint_path $checkpoint_path"
    fi

    echo $command
    eval $command
  done
done


echo "All evaluations completed!"
