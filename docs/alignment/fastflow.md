# FastFlow implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_optional_dependency</code> — The canonical implementation imports or requires a project extra that a core installation does not provide.
- **Method family:** Normalizing flow
- **Registry entry:** <code>FastFlowDetector</code>
- **Detector module:** <code>baoiad.models.detectors.fastflow</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2111.07677](https://arxiv.org/abs/2111.07677)
- **Source repository:** [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- **Source revision:** [4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a](https://github.com/open-edge-platform/anomalib/commit/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a)
- **Config README:** [configs/fastflow/README.md](../../configs/fastflow/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Repository-local implementation with MMEngine integration.

## Limitations

- FrEIA is an optional project extra but FastFlow imports it unconditionally, so the canonical detector cannot be imported from a core-only installation.
- The current round did not publish a fresh complete 15-category raw result artifact.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
