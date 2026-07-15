# DRAEM implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Self-supervised synthesis
- **Registry entry:** <code>DRAEMDetector</code>
- **Detector module:** <code>baoiad.models.detectors.draem</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2108.07610](https://arxiv.org/abs/2108.07610)
- **Source repository:** [https://github.com/VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM)
- **Source revision:** [2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef](https://github.com/VitjanZ/DRAEM/commit/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef)
- **Config README:** [configs/draem/README.md](../../configs/draem/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Repository-local reimplementation with MMEngine data synthesis, loss, and evaluation adapters.

## Limitations

- Referenced raw validation artifacts are not distributed, so the alignment narrative is not independently verifiable from a public clone.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
