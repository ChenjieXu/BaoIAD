# MuSc implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>MuScDetector</code>
- **Detector module:** <code>baoiad.models.detectors.musc</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2401.16753](https://arxiv.org/abs/2401.16753)
- **Source repository:** [https://github.com/xrli-U/MuSc](https://github.com/xrli-U/MuSc)
- **Source revision:** [72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4](https://github.com/xrli-U/MuSc/commit/72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4)
- **Config README:** [configs/musc/README.md](../../configs/musc/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MIT-licensed LNAMD, MSM, and RsCIN sections are explicitly marked copied and modified; MMEngine and batched-scoring adapters are added. PatchMaker-related secondary attribution remains to be frozen.

## Limitations

- Copied sections require source-license and modification-notice review.
- The canonical strict configuration requires a non-distributed local OpenCLIP reference tree and fails when it is absent.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
