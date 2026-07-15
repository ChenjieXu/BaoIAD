# SimpleNet implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Discriminative
- **Registry entry:** <code>SimpleNetDetector</code>
- **Detector module:** <code>baoiad.models.detectors.simplenet</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2303.15140](https://arxiv.org/abs/2303.15140)
- **Source repository:** [https://github.com/DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)
- **Source revision:** [351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b](https://github.com/DonaldRR/SimpleNet/commit/351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b)
- **Config README:** [configs/simplenet/README.md](../../configs/simplenet/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

MMEngine integration and repository-local optimizer, post-processing, and evaluation adapters.

## Limitations

- Referenced raw validation artifacts are not distributed, so the alignment narrative is not independently verifiable from a public clone.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
