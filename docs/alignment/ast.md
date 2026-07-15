# AST implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Knowledge distillation
- **Registry entry:** <code>ASTDetector</code>
- **Detector module:** <code>baoiad.models.detectors.ast</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2210.07829](https://arxiv.org/abs/2210.07829)
- **Source repository:** [https://github.com/marco-rudolph/AST](https://github.com/marco-rudolph/AST)
- **Source revision:** [8c243ad9adac68e874f87edc6618aa5ea2827228](https://github.com/marco-rudolph/AST/commit/8c243ad9adac68e874f87edc6618aa5ea2827228)
- **Config README:** [configs/ast/README.md](../../configs/ast/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Closely derived flow, positional-encoding, teacher, and student structures plus MMEngine integration; the frozen upstream tree has no identifiable redistribution license.

## Limitations

- Redistribution is blocked pending written permission or a documented clean-room replacement.
- The alignment record retains an unresolved toothbrush image-level gap.
- Referenced raw validation artifacts are not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
