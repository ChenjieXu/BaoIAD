#!/bin/bash
# BaoIAD environment configuration
# Source this file before running tools: source tools/env.sh

# Data root - override this to point to your dataset directory
export BAOIAD_DATA_ROOT="${BAOIAD_DATA_ROOT:-$(dirname "$0")/../data}"

# Cache directories
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# Optional: use HF mirror (set to 1 to enable)
# export BAOIAD_USE_MIRROR=1

# Optional: cache dir for model weights
# export BAOIAD_CACHE_DIR="/dev/shm/baoiad-cache"
