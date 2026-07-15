# AACLIP implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>AACLIPDetector</code>
- **Detector module:** <code>baoiad.models.detectors.aaclip</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2503.06661](https://arxiv.org/abs/2503.06661)
- **Source repository:** [https://github.com/Mwxinnn/AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)
- **Source revision:** [53db195f230442aa118c246876c94ba1c76139cc](https://github.com/Mwxinnn/AA-CLIP/commit/53db195f230442aa118c246876c94ba1c76139cc)
- **Config README:** [configs/aaclip/README.md](../../configs/aaclip/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and repository-local adapter/checkpoint paths.

## Limitations

- The official target adapter checkpoint is not distributed, Stage 2 numerical comparison remains open, and canonical config requires absent local reference assets.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
