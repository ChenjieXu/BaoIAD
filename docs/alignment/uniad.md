# UniAD implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Reconstruction / ViT
- **Registry entry:** <code>UniADDetector</code>
- **Detector module:** <code>baoiad.models.detectors.uniad_detector</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2206.03687](https://arxiv.org/abs/2206.03687)
- **Source repository:** [https://github.com/zhangzjn/ADer](https://github.com/zhangzjn/ADer)
- **Source revision:** [902937a7ed7fa7689674a4ac9b8fe9a72a40c402](https://github.com/zhangzjn/ADer/commit/902937a7ed7fa7689674a4ac9b8fe9a72a40c402)
- **Config README:** [configs/uniad/README.md](../../configs/uniad/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

MMEngine implementation aligned to ADer; source attribution must be revalidated against the original Apache-2.0 UniAD repository and any ADer-only changes isolated.

## Limitations

- ADer-derived details remain on legal hold until dual-source traceability is complete.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
