# MemSeg implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Reconstruction / ViT
- **Registry entry:** <code>MemSegDetector</code>
- **Detector module:** <code>baoiad.models.detectors.memseg</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2205.00908](https://arxiv.org/abs/2205.00908)
- **Source repository:** [https://github.com/TooTouch/MemSeg](https://github.com/TooTouch/MemSeg)
- **Source revision:** [836bd465a9b14422f92666dc29dc36edce2692d0](https://github.com/TooTouch/MemSeg/commit/836bd465a9b14422f92666dc29dc36edce2692d0)
- **Config README:** [configs/memseg/README.md](../../configs/memseg/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Repository-local reimplementation with MMEngine integration, Coordinate Attention, memory, and data-synthesis adapters.

## Limitations

- The revised strict 15-category path has not been freshly rerun, and cross-framework RNG trajectory differences remain.
- Referenced raw evidence is not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
