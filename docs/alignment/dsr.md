# DSR implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>network_dependent</code> — The canonical path may download a model or other external artifact when it is absent locally.
- **Method family:** Self-supervised synthesis
- **Registry entry:** <code>DSRDetector</code>
- **Detector module:** <code>baoiad.models.detectors.dsr</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2208.01521](https://arxiv.org/abs/2208.01521)
- **Source repository:** [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- **Source revision:** [4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a](https://github.com/open-edge-platform/anomalib/commit/4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a)
- **Config README:** [configs/dsr/README.md](../../configs/dsr/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Repository-local reimplementation with MMEngine integration and references to external pretrained artifacts.

## Limitations

- The canonical auto setting downloads pretrained VQ-VAE weights when they are absent locally.
- Referenced raw evidence and specific weights are not distributed and require provenance review.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
