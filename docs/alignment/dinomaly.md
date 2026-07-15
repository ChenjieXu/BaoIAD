# Dinomaly implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>network_dependent</code> — The canonical path may download a model or other external artifact when it is absent locally.
- **Method family:** Reconstruction / ViT
- **Registry entry:** <code>DinomalyDetector</code>
- **Detector module:** <code>baoiad.models.detectors.dinomaly</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2405.14325](https://arxiv.org/abs/2405.14325)
- **Source repository:** [https://github.com/guojiajeremy/Dinomaly](https://github.com/guojiajeremy/Dinomaly)
- **Source revision:** [c5c76d01a2bd7212f1c4b7dfdad14902d0f48cfe](https://github.com/guojiajeremy/Dinomaly/commit/c5c76d01a2bd7212f1c4b7dfdad14902d0f48cfe)
- **Config README:** [configs/dinomaly/README.md](../../configs/dinomaly/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and repository-local configuration/evaluation adapters.

## Limitations

- The encoder downloads DINOv2 weights when they are absent locally.
- Referenced raw validation artifacts are not distributed, so the alignment narrative is not independently verifiable from a public clone.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
