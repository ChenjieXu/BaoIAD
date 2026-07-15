# NSA implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Self-supervised synthesis
- **Registry entry:** <code>NSADetector</code>
- **Detector module:** <code>baoiad.models.detectors.nsa</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2109.15222](https://arxiv.org/abs/2109.15222)
- **Source repository:** [https://github.com/hmsch/natural-synthetic-anomalies](https://github.com/hmsch/natural-synthetic-anomalies)
- **Source revision:** [919591685307ce030fe27cb77687509dc277189c](https://github.com/hmsch/natural-synthetic-anomalies/commit/919591685307ce030fe27cb77687509dc277189c)
- **Config README:** [configs/nsa/README.md](../../configs/nsa/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Repository-local port of synthesis and training semantics with MMEngine integration; the release inventory pins an audit reference even though the historical derivation revision was not recorded.

## Limitations

- The exact historical derivation revision was not recorded; the release inventory pins the current upstream audit tree.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
