#!/bin/bash
# Smoke test for remaining 12 methods + 5 timeout methods
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RESULTS_DIR="runs/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

METHODS="csflow draem dsr fastflow patchcore rdpp regad simplenet spade stfpm supersimplenet uflow uniad uninet vitad winclip"

SUMMARY="$RESULTS_DIR/summary.txt"
echo "IADBench Smoke Test - $(date)" > "$SUMMARY"
echo "========================================" >> "$SUMMARY"

for method in $METHODS; do
    # Find config
    cfg=$(ls configs/$method/*256*mvtec*.py 2>/dev/null | grep -v unified | grep -v bak | head -1)
    if [ -z "$cfg" ]; then
        cfg=$(ls configs/$method/*.py 2>/dev/null | grep -v unified | grep -v bak | head -1)
    fi
    workdir="$RESULTS_DIR/$method"
    mkdir -p "$workdir"

    echo ""
    echo "=== [$method] ==="
    echo "Config: $cfg"

    if [ -z "$cfg" ] || [ ! -f "$cfg" ]; then
        echo "[$method] SKIP - config not found"
        echo "$method: SKIP (config not found)" >> "$SUMMARY"
        continue
    fi

    t_start=$(date +%s)
    timeout 600 $PYTHON tools/train.py "$cfg" \
        --work-dir "$workdir" \
        --cfg-options train_cfg.max_epochs=2 train_cfg.val_interval=1 \
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
    echo "$method: $status (${elapsed}s)" >> "$SUMMARY"
    grep -i "auroc\|aupro" "$workdir.log" | tail -3 >> "$SUMMARY" 2>/dev/null
done

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "Done at $(date)" >> "$SUMMARY"
cat "$SUMMARY"
