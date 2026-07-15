# PyramidFlow implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Normalizing flow
- **Registry entry:** <code>PyramidFlowDetector</code>
- **Detector module:** <code>baoiad.models.detectors.pyramidflow</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2303.02595](https://arxiv.org/abs/2303.02595)
- **Source repository:** [https://github.com/gasharper/PyramidFlow](https://github.com/gasharper/PyramidFlow)
- **Source revision:** Not recorded in the release inventory.
- **Config README:** [configs/pyramidflow/README.md](../../configs/pyramidflow/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Core implementation is closely derived from an ADer proxy snapshot whose fixed tree has no identifiable redistribution license; the official repository is unavailable.

## Limitations

- Closure is proxy-only, official source is unavailable, speed is unavailable, and redistribution is on legal hold.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
