# DeSTSeg implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Knowledge distillation
- **Registry entry:** <code>DeSTSegDetector</code>
- **Detector module:** <code>baoiad.models.detectors.destseg</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2211.11317](https://arxiv.org/abs/2211.11317)
- **Source repository:** [https://github.com/apple/ml-destseg](https://github.com/apple/ml-destseg)
- **Source revision:** [f6ea31fb5b097698b195f85b1d5e3efaedce9eb6](https://github.com/apple/ml-destseg/commit/f6ea31fb5b097698b195f85b1d5e3efaedce9eb6)
- **Config README:** [configs/destseg/README.md](../../configs/destseg/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration; model and transform sections follow Apple sample code whose custom license, trademark terms, and lack of patent grant require legal review.

## Limitations

- The canonical strict configuration resolves dtd_path='auto' only from a local data/dtd tree and raises when DTD images are absent; those assets are not distributed.
- The corrected mainline is still validation-in-progress and the recorded fresh rerun covers only 10 of 15 categories.
- The config README's full result summary must not be presented as fresh corrected-mainline evidence.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
