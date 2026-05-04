#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the repo-local 37-method inventory.
ALL_METHODS="glass dinomaly simplenet rdpp ast rd uninet supersimplenet vitad uflow efficientad patchcore destseg musc memseg anomalydino cflow draem padim cfa aaclip differnet dfm fastflow uniad anovl anomalyclip dsr winclip regad cutpaste pyramidflow nsa adaclip dfkde saaplus ganomaly"
METHODS="$ALL_METHODS"
SUMMARY="runs/smoke_test_summary.txt"
mkdir -p runs
: > "$SUMMARY"

if [ "$#" -gt 0 ]; then
  METHODS="$*"
fi

for method in $METHODS; do
  echo "=== [$method] smoke ==="
  cfg=$(python3 - <<PY
from tools import benchmark
print(benchmark.find_config('$method') or '')
PY
)
  if [ -z "$cfg" ]; then
    echo "$method: missing config" | tee -a "$SUMMARY"
    exit 1
  fi
  echo "$method: $cfg" | tee -a "$SUMMARY"
done
