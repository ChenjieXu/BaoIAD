# AnomalyDINO implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Few-shot / registration
- **Registry entry:** <code>AnomalyDINODetector</code>
- **Detector module:** <code>baoiad.models.detectors.anomalydino</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2405.14529](https://arxiv.org/abs/2405.14529)
- **Source repository:** [https://github.com/dammsi/AnomalyDINO](https://github.com/dammsi/AnomalyDINO)
- **Source revision:** [b9d1c2648e3a5247437d4d953d907a8f3d994457](https://github.com/dammsi/AnomalyDINO/commit/b9d1c2648e3a5247437d4d953d907a8f3d994457)
- **Config README:** [configs/anomalydino/README.md](../../configs/anomalydino/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and equivalent normalized-dot-product distance in place of the reference FAISS expression.

## Limitations

- The detector is conditionally imported.
- The alignment record says the unified probe CLI remains absent.
- Referenced raw validation artifacts are not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
