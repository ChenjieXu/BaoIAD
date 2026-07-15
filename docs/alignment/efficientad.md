# EfficientAD implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>network_dependent</code> — The canonical path may download a model or other external artifact when it is absent locally.
- **Method family:** Knowledge distillation
- **Registry entry:** <code>EfficientADDetector</code>
- **Detector module:** <code>baoiad.models.detectors.efficientad</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2303.14535](https://arxiv.org/abs/2303.14535)
- **Source repository:** [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- **Source revision:** [4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a](https://github.com/open-edge-platform/anomalib/commit/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a)
- **Config README:** [configs/efficientad/README.md](../../configs/efficientad/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Repository-local reimplementation with MMEngine integration and explicit references to external pretrained artifacts.

## Limitations

- The alignment record says an official-pretrained 15-category run remains incomplete.
- The detector downloads the referenced pretrained teacher when it is absent locally; its artifact license and checksum are not frozen.
- Referenced raw evidence and weights are not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
