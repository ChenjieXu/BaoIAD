#!/usr/bin/env bash
set -euo pipefail

# Remaining-method helper now follows the repo-local inventory.
METHODS="glass dinomaly simplenet rdpp ast rd uninet supersimplenet vitad uflow efficientad patchcore destseg musc memseg anomalydino cflow draem padim cfa aaclip differnet dfm fastflow uniad anovl anomalyclip dsr winclip regad cutpaste pyramidflow nsa adaclip dfkde saaplus ganomaly"
for method in $METHODS; do
  python3 - <<PY
from tools import benchmark
assert benchmark.find_config('$method'), '$method'
print('$method: ok')
PY
done
