# BaoIAD repository guide

BaoIAD is a self-contained industrial anomaly detection benchmark repository. The repo-local method inventory is maintained in [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py), and `python tools/benchmark.py --methods all` selects those 37 method slugs.

For method details, use the config-local READMEs under [`configs/`](../../configs/). For implementation-alignment process notes, use [`docs/alignment/`](../alignment/).
