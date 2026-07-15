# GLASS implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Self-supervised synthesis
- **Registry entry:** <code>GLASSDetector</code>
- **Detector module:** <code>baoiad.models.detectors.glass</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2407.09359](https://arxiv.org/abs/2407.09359)
- **Source repository:** [https://github.com/cqylunlun/GLASS](https://github.com/cqylunlun/GLASS)
- **Source revision:** [6af03b9d7f7b33a1aebd69cd4c30a41bf020a2d1](https://github.com/cqylunlun/GLASS/commit/6af03b9d7f7b33a1aebd69cd4c30a41bf020a2d1)
- **Config README:** [configs/glass/README.md](../../configs/glass/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

MMEngine integration and repository-local data, loop, optimizer, and evaluator adapters.

## Limitations

- The recorded strict benchmark covers 14 of 15 MVTec categories; screw remains incomplete.
- The canonical strict configuration requires non-distributed foreground masks, a distribution spreadsheet, and DTD data.
- The GLASS model and dataset import pandas unconditionally, but pandas is not declared as a core or method-specific project dependency.
- Referenced raw evidence is not distributed in the public repository.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
