# WinCLIP implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>network_dependent</code> — The canonical path may download a model or other external artifact when it is absent locally.
- **Method family:** Vision-language / foundation
- **Registry entry:** <code>WinClipDetector</code>
- **Detector module:** <code>baoiad.models.detectors.winclip</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2303.14814](https://arxiv.org/abs/2303.14814)
- **Source repository:** [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- **Source revision:** Not recorded in the release inventory.
- **Config README:** [configs/winclip/README.md](../../configs/winclip/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json) · [compliance checker](../../tools/check_release_compliance.py)

## Implementation differences

Repository-local reimplementation ported from anomalib with MMEngine integration.

## Limitations

- The alignment record has no readable frozen revision, and missing local weights fall back to network access.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
