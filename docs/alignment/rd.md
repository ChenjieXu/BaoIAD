# RD implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Knowledge distillation
- **Registry entry:** <code>ReverseDistillation</code>
- **Detector module:** <code>baoiad.models.detectors.reverse_distillation</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2201.10703](https://arxiv.org/abs/2201.10703)
- **Source repository:** [https://github.com/hq-deng/RD4AD](https://github.com/hq-deng/RD4AD)
- **Source revision:** [6554076872c65f8784f6ece8cfb39ce77e1aee12](https://github.com/hq-deng/RD4AD/commit/6554076872c65f8784f6ece8cfb39ce77e1aee12)
- **Config README:** [configs/rd/README.md](../../configs/rd/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

MMEngine integration and repository-local feature, loss, and evaluator adapters.

## Limitations

- Referenced raw validation artifacts are not distributed, so the alignment narrative is not independently verifiable from a public clone.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
