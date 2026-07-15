# PaDiM implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Feature-memory / density
- **Registry entry:** <code>PaDiMDetector</code>
- **Detector module:** <code>baoiad.models.detectors.padim</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2011.08785](https://arxiv.org/abs/2011.08785)
- **Source repository:** [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- **Source revision:** [0ef8ab1e43340bddf4d92d1f046c3d34a83af6b0](https://github.com/open-edge-platform/anomalib/commit/0ef8ab1e43340bddf4d92d1f046c3d34a83af6b0)
- **Config README:** [configs/padim/README.md](../../configs/padim/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

MMEngine implementation audited against the expanded public anomalib revision recorded by the alignment snapshot.

## Limitations

- The public source revision is frozen, but referenced raw validation evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
