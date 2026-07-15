# CutPaste implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Self-supervised synthesis
- **Registry entry:** <code>CutPasteDetector</code>
- **Detector module:** <code>baoiad.models.detectors.cutpaste</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2104.04015](https://arxiv.org/abs/2104.04015)
- **Source repository:** [https://github.com/Runinho/pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)
- **Source revision:** [10d8bf71df76d3a97f0106efee1d76f81d983149](https://github.com/Runinho/pytorch-cutpaste/commit/10d8bf71df76d3a97f0106efee1d76f81d983149)
- **Config README:** [configs/cutpaste/README.md](../../configs/cutpaste/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Closely derived CutPaste/Scar augmentation, projection head, and Gaussian-density scoring logic with tensor and MMEngine adapters; the frozen upstream tree has no identifiable redistribution license.

## Limitations

- Redistribution is blocked pending written permission or a documented clean-room replacement.
- The alignment record leaves corruption severity, root cause, and the final mainline choice unresolved.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
