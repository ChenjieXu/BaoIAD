# SuperSimpleNet implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Discriminative
- **Registry entry:** <code>SuperSimpleNetDetector</code>
- **Detector module:** <code>baoiad.models.detectors.supersimplenet</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2408.03143](https://arxiv.org/abs/2408.03143)
- **Source repository:** [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- **Source revision:** [4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a](https://github.com/open-edge-platform/anomalib/commit/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a)
- **Config README:** [configs/supersimplenet/README.md](../../configs/supersimplenet/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Repository-local reimplementation with MMEngine integration and modified image-score handling.

## Limitations

- The alignment record says the strict 300-epoch 15-category rerun and image-score confirmation remain open.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
