# AnomalyCLIP implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_undistributed_assets</code> — The canonical path requires local assets, checkpoints, support sets, or datasets that are not distributed with the repository.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>AnomalyCLIPOfficialDetector</code>
- **Detector module:** <code>baoiad.models.detectors.anomalyclip_official</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2310.18961](https://arxiv.org/abs/2310.18961)
- **Source repository:** [https://github.com/zqhang/AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP)
- **Source revision:** [3911738c0867544f545a076ad78f3f11d9ecbfdf](https://github.com/zqhang/AnomalyCLIP/commit/3911738c0867544f545a076ad78f3f11d9ecbfdf)
- **Config README:** [configs/anomalyclip/README.md](../../configs/anomalyclip/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

MMEngine integration; the public inventory selects the official detector while a legacy detector class remains registered.

## Limitations

- Canonical config requires absent local reference/AnomalyCLIP assets.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
