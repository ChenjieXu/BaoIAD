# DifferNet implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Normalizing flow
- **Registry entry:** <code>DifferNetDetector</code>
- **Detector module:** <code>baoiad.models.detectors.differnet</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2008.12577](https://arxiv.org/abs/2008.12577)
- **Source repository:** [https://github.com/marco-rudolph/differnet](https://github.com/marco-rudolph/differnet)
- **Source revision:** [9bdf02686297a093fb206ffeba64b1c0e78182b6](https://github.com/marco-rudolph/differnet/commit/9bdf02686297a093fb206ffeba64b1c0e78182b6)
- **Config README:** [configs/differnet/README.md](../../configs/differnet/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Closely derived flow subnet, permutation, coupling, and density logic with MMEngine integration; the frozen upstream tree has no identifiable redistribution license.

## Limitations

- Redistribution is blocked pending written permission or a documented clean-room replacement.
- The method is image-level; any uniform pixel map is only API compatibility output.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
