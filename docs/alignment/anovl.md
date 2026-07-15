# AnoVL implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>AnoVLDetector</code>
- **Detector module:** <code>baoiad.models.detectors.anovl</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2308.15939](https://arxiv.org/abs/2308.15939)
- **Source repository:** [https://github.com/hq-deng/AnoVL](https://github.com/hq-deng/AnoVL)
- **Source revision:** [3a70bfdaea6baf1eeb140c5de8155b535bd94833](https://github.com/hq-deng/AnoVL/commit/3a70bfdaea6baf1eeb140c5de8155b535bd94833)
- **Config README:** [configs/anovl/README.md](../../configs/anovl/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and repository-local OpenCLIP checkpoint path.

## Limitations

- Canonical configuration points to an absent local OpenCLIP checkpoint.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
