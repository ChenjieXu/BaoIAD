#!/bin/bash
# IADBench Smoke Test - GPU version (RTX 3090 24GB)
# Usage: bash tools/smoke_test_gpu.sh [--all | method1 method2 ...]
# Default: run all methods

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
RESULTS_DIR="runs/smoke_gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# All methods
ALL_METHODS="aaclip anomalyclip ast cfa cflow csflow cutpaste destseg dfkde dfm differnet dinomaly draem dsr efficientad fastflow fre ganomaly graphcore invad memae musc nsa padim patchcore pyramidflow rd rdpp regad simplenet spade stfpm supersimplenet uflow uniad uninet vitad winclip"

# Methods that need small batch size on 24GB GPU
TINY_BS="uniad"  # bs=1 (OOM at bs=2 due to transformer attention maps)
SMALL_BS="spade winclip"  # bs=2
MEDIUM_BS="ast patchcore uninet regad"  # bs=4

# Methods that need longer timeout (memory bank construction is slow)
LONG_TIMEOUT="patchcore graphcore anomalyclip"  # 1800s instead of 900s
DEFAULT_TIMEOUT=900

if [ "$1" = "--all" ] || [ $# -eq 0 ]; then
    METHODS="$ALL_METHODS"
else
    METHODS="$@"
fi

SUMMARY="$RESULTS_DIR/summary.txt"
echo "IADBench GPU Smoke Test - $(date)" > "$SUMMARY"
echo "========================================" >> "$SUMMARY"

get_batch_size() {
    local m="$1"
    for s in $TINY_BS; do [ "$m" = "$s" ] && echo 1 && return; done
    for s in $SMALL_BS; do [ "$m" = "$s" ] && echo 2 && return; done
    for s in $MEDIUM_BS; do [ "$m" = "$s" ] && echo 4 && return; done
    echo 8
}

get_timeout() {
    local m="$1"
    for s in $LONG_TIMEOUT; do [ "$m" = "$s" ] && echo 1800 && return; done
    echo $DEFAULT_TIMEOUT
}

for method in $METHODS; do
    if [ "$method" = "memae" ]; then
        echo ""
        echo "=== [memae] SKIP ==="
        echo "[memae] SKIP - official strict alignment uses video datasets and dedicated configs/scripts"
        echo "memae: SKIP (use tools/memae_official_video_smoke.sh)" >> "$SUMMARY"
        continue
    fi

    if [ "$method" = "cflow" ] && [ -f configs/cflow/cflow_mvtec_strict.py ]; then
        cfg="configs/cflow/cflow_mvtec_strict.py"
    else
        cfg=$(ls configs/$method/*256*mvtec*.py 2>/dev/null | grep -v unified | grep -v bak | head -1)
        if [ -z "$cfg" ]; then
            cfg=$(ls configs/$method/*.py 2>/dev/null | grep -v unified | grep -v bak | head -1)
        fi
    fi
    workdir="$RESULTS_DIR/$method"
    mkdir -p "$workdir"

    if [ -z "$cfg" ] || [ ! -f "$cfg" ]; then
        echo "[$method] SKIP - config not found"
        echo "$method: SKIP (config not found)" >> "$SUMMARY"
        continue
    fi

    bs=$(get_batch_size "$method")
    method_timeout=$(get_timeout "$method")
    echo ""
    echo "=== [$method] bs=$bs timeout=${method_timeout}s ==="

    # Detect iter-based configs (by_epoch=False) — use max_iters instead of max_epochs
    if grep -q 'by_epoch=False' "$cfg" 2>/dev/null; then
        TRAIN_CFG_OPTS="train_cfg.max_iters=20 train_cfg.val_interval=10"
    else
        TRAIN_CFG_OPTS="train_cfg.max_epochs=2 train_cfg.val_interval=1"
    fi

    t_start=$(date +%s)
    timeout $method_timeout $PYTHON tools/train.py "$cfg" \
        --work-dir "$workdir" \
        --cfg-options \
            $TRAIN_CFG_OPTS \
            train_dataloader.batch_size=$bs \
            test_dataloader.batch_size=$bs \
            val_dataloader.batch_size=$bs \
            "train_dataloader.dataset.cls_names=['bottle']" \
            "test_dataloader.dataset.cls_names=['bottle']" \
            "val_dataloader.dataset.cls_names=['bottle']" \
            train_dataloader.dataset.multi_class=False \
            test_dataloader.dataset.multi_class=False \
            val_dataloader.dataset.multi_class=False \
        > "$workdir.log" 2>&1
    exit_code=$?
    t_end=$(date +%s)
    elapsed=$((t_end - t_start))

    if [ $exit_code -eq 0 ]; then
        status="PASS"
    elif [ $exit_code -eq 124 ]; then
        status="TIMEOUT"
    else
        status="FAIL (exit $exit_code)"
    fi

    echo "[$method] $status (${elapsed}s)"
    echo "$method: $status (${elapsed}s, bs=$bs)" >> "$SUMMARY"
    grep -i "image_auroc.*pixel_auroc" "$workdir.log" | tail -1 >> "$SUMMARY" 2>/dev/null
done

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "Done at $(date)" >> "$SUMMARY"
echo ""
echo "=== SUMMARY ==="
cat "$SUMMARY"
