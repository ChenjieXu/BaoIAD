# GANomaly implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Reconstruction / ViT
- **Registry entry:** <code>GanomalyDetector</code>
- **Detector module:** <code>baoiad.models.detectors.ganomaly</code>

## Public references

- **Paper:** [https://arxiv.org/abs/1805.06725](https://arxiv.org/abs/1805.06725)
- **Source repository:** [https://github.com/samet-akcay/ganomaly](https://github.com/samet-akcay/ganomaly)
- **Source revision:** [78da4ea9a99f5b02ab60dd651a18def929176d77](https://github.com/samet-akcay/ganomaly/commit/78da4ea9a99f5b02ab60dd651a18def929176d77)
- **Config README:** [configs/ganomaly/README.md](../../configs/ganomaly/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Repository-local encoder-decoder-encoder implementation with MMEngine integration.

## Limitations

- The current strict path supports image-level evaluation only; it must not be presented as having comparable pixel localization evidence.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
