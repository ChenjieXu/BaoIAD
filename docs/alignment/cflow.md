# CFlow implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Normalizing flow
- **Registry entry:** <code>CFlowDetector</code>
- **Detector module:** <code>baoiad.models.detectors.cflow</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2107.12571](https://arxiv.org/abs/2107.12571)
- **Source repository:** [https://github.com/gudovskiy/cflow-ad](https://github.com/gudovskiy/cflow-ad)
- **Source revision:** [b2ebf9e673a0aa46992a3b18367ec066a57bba89](https://github.com/gudovskiy/cflow-ad/commit/b2ebf9e673a0aa46992a3b18367ec066a57bba89)
- **Config README:** [configs/cflow/README.md](../../configs/cflow/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Repository-local FrEIA implementation with MMEngine transforms and a strict guard for a non-distributed local reference tree.

## Limitations

- Canonical strict configuration requires absent local reference/cflow-ad assets.
- FrEIA is an optional project extra; without it CFlow substitutes a lightweight fallback that is not the declared official-alignment implementation.
- The alignment record says the full archived benchmark item remains incomplete.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
