# U-Flow implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_optional_dependency</code> — The canonical implementation imports or requires a project extra that a core installation does not provide.
- **Method family:** Normalizing flow
- **Registry entry:** <code>UFlowDetector</code>
- **Detector module:** <code>baoiad.models.detectors.uflow</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2211.12353](https://arxiv.org/abs/2211.12353)
- **Source repository:** [https://github.com/mtailanian/uflow](https://github.com/mtailanian/uflow)
- **Source revision:** [d6217844836790773f2c4b91ff3046c59b23f027](https://github.com/mtailanian/uflow/commit/d6217844836790773f2c4b91ff3046c59b23f027)
- **Config README:** [configs/uflow/README.md](../../configs/uflow/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Contains anomalib-derived ported blocks and an optional NFA path closely derived from AGPL-3.0 upstream code; default strict config disables NFA.

## Limitations

- FrEIA is an optional project extra but U-Flow imports it unconditionally, so the canonical detector cannot be imported from a core-only installation.
- The AGPL-derived NFA implementation must be removed or clean-room replaced before Apache-only release.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
