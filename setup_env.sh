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

conda activate diffusion


# Set swanlab project

export SWANLAB_PROJECT=llada_sft