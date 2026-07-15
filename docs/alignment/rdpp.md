# RD++ implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>blocked_by_optional_dependency</code> — The canonical implementation imports or requires a project extra that a core installation does not provide.
- **Method family:** Knowledge distillation
- **Registry entry:** <code>RDPPDetector</code>
- **Detector module:** <code>baoiad.models.detectors.rdpp</code>

## Public references

- **Paper:** [https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html](https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html)
- **Source repository:** [https://github.com/tientrandinh/Revisiting-Reverse-Distillation](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
- **Source revision:** [7f2ceb7c87e602617b8600e1a498f7ef7f5247d6](https://github.com/tientrandinh/Revisiting-Reverse-Distillation/commit/7f2ceb7c87e602617b8600e1a498f7ef7f5247d6)
- **Config README:** [configs/rdpp/README.md](../../configs/rdpp/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

MMEngine integration plus a vendored MIT-licensed simplex-noise helper that still needs RD++ and OpenSimplex attribution details.

## Limitations

- The canonical strict configuration requires the optional geomloss package for the official Sinkhorn SSOT loss and fails closed when it is absent.
- Referenced raw validation artifacts are not distributed, so the alignment narrative is not independently verifiable from a public clone.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
