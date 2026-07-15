#!/bin/bash
# BaoIAD environment configuration
# Source this file before running tools: source tools/env.sh

# Data root - override this to point to your dataset directory.
# BASH_SOURCE is stable when this file is sourced from another directory.
_BAOIAD_ENV_FILE="${BASH_SOURCE[0]:-$0}"
_BAOIAD_REPO_ROOT="$(cd "$(dirname "$_BAOIAD_ENV_FILE")/.." && pwd -P)"
export BAOIAD_DATA_ROOT="${BAOIAD_DATA_ROOT:-$_BAOIAD_REPO_ROOT/data}"
unset _BAOIAD_ENV_FILE _BAOIAD_REPO_ROOT

# Cache directories
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# Optional: use HF mirror (set to 1 to enable)
# export BAOIAD_USE_MIRROR=1

# Optional: cache dir for model weights
# export BAOIAD_CACHE_DIR="/dev/shm/baoiad-cache"
