#!/bin/bash

# Set proxy configuration
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# Set Hugging Face cache directory
export HF_HOME=/cephfs/shared/hf_cache/

# Set required environment variables for evaluation
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "Environment variables set:"
echo "Proxy: $http_proxy"
echo "HF_HOME: $HF_HOME"
echo "HF_ALLOW_CODE_EVAL: $HF_ALLOW_CODE_EVAL"
echo "HF_DATASETS_TRUST_REMOTE_CODE: $HF_DATASETS_TRUST_REMOTE_CODE" 

# Set swanlab project
export SWANLAB_PROJECT=llada_sft



model_name="GSAI-ML/LLaDA-8B-Instruct"
batch_size=2
learning_rate=1e-5
grad_accum_steps=4
output_dir="./outputs"
job_name="llada-s1"

# train_data="simplescaling/s1K"

train_data="zwhe99/DeepMath-103K"
max_train_samples=10000
num_epochs=3


# cluster 
gpus=8

command="accelerate launch --config_file ddp_config.yaml --main_process_port 29500 --num_processes $gpus sft_train.py \
    --model_name $model_name \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --grad_accum_steps $grad_accum_steps \
    --output_dir $output_dir \
    --job_name $job_name \
    --train_data $train_data \
    --use_fast_preprocessing"

if [ -n "$max_train_samples" ]; then
    command="$command --max_train_samples $max_train_samples"
fi

echo $command
eval $command