# AdaCLIP implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>AdaCLIPDetector</code>
- **Detector module:** <code>baoiad.models.detectors.adaclip</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2407.15795](https://arxiv.org/abs/2407.15795)
- **Source repository:** [https://github.com/caoyunkang/AdaCLIP](https://github.com/caoyunkang/AdaCLIP)
- **Source revision:** [b762ac40c3f33c77e7e513e48cb436f059d456da](https://github.com/caoyunkang/AdaCLIP/commit/b762ac40c3f33c77e7e513e48cb436f059d456da)
- **Config README:** [configs/adaclip/README.md](../../configs/adaclip/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Repository-local implementation following the official prompt and adapter design with MMEngine integration.

## Limitations

- Official checkpoints are external, and referenced raw validation artifacts are not distributed.
- The alignment record reports large per-category gaps and a non-official MVTec-only fallback path that remains unresolved.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
