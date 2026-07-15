# RegAD implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Few-shot / registration
- **Registry entry:** <code>RegADDetector</code>
- **Detector module:** <code>baoiad.models.detectors.regad</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2207.07361](https://arxiv.org/abs/2207.07361)
- **Source repository:** [https://github.com/MediaBrain-SJTU/RegAD](https://github.com/MediaBrain-SJTU/RegAD)
- **Source revision:** [5e2c1f8c18d302b0354471567846fee3ed2ff063](https://github.com/MediaBrain-SJTU/RegAD/commit/5e2c1f8c18d302b0354471567846fee3ed2ff063)
- **Config README:** [configs/regad/README.md](../../configs/regad/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and explicit support-set handling; despite the strict filename, the canonical configuration permits deterministic local support sampling when the official support set is absent.

## Limitations

- The canonical strict configuration sets strict_require_official_support_set=False, so the runner silently uses a non-official deterministic local support-sampling fallback when the official support set is absent.
- Official support-set data is external, and referenced raw validation evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
