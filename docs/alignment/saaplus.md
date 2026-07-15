# SAA+ implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>network_dependent</code> — The canonical path may download a model or other external artifact when it is absent locally.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>SAADetector</code>
- **Detector module:** <code>baoiad.models.detectors.saa</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2305.10724](https://arxiv.org/abs/2305.10724)
- **Source repository:** [https://github.com/caoyunkang/Segment-Any-Anomaly](https://github.com/caoyunkang/Segment-Any-Anomaly)
- **Source revision:** [ff564ed09bef91d86452f62aa1564e778580513e](https://github.com/caoyunkang/Segment-Any-Anomaly/commit/ff564ed09bef91d86452f62aa1564e778580513e)
- **Config README:** [configs/saaplus/README.md](../../configs/saaplus/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and ported MVTec prompt definitions.

## Limitations

- Canonical configuration references three absent pretrained files and may implicitly download multi-gigabyte external weights.
- The detector is conditionally imported and raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
