# ViTAD implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Partially verified
- **Public validation evidence distributed:** No
- **Runtime state:** <code>network_dependent</code> — The canonical path may download a model or other external artifact when it is absent locally.
- **Method family:** Reconstruction / ViT
- **Registry entry:** <code>ViTADDetector</code>
- **Detector module:** <code>baoiad.models.detectors.vitad</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2312.07495](https://arxiv.org/abs/2312.07495)
- **Source repository:** [https://github.com/zhangzjn/ADer](https://github.com/zhangzjn/ADer)
- **Source revision:** [902937a7ed7fa7689674a4ac9b8fe9a72a40c402](https://github.com/zhangzjn/ADer/commit/902937a7ed7fa7689674a4ac9b8fe9a72a40c402)
- **Config README:** [configs/vitad/README.md](../../configs/vitad/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Closely derived ADer fusion, decoder, backbone, loss, and metric structures plus MMEngine integration; the frozen ADer tree has no identifiable redistribution license.

## Limitations

- Redistribution is blocked pending written permission or a documented clean-room replacement.
- The alignment table still records an open implementation-equivalence item that requires reconciliation.
- The official per-epoch order artifact is not distributed; `train_vitad_exact_order.py` only replays a user-supplied, independently verified JSON and stops with an actionable error when it is absent.
- The canonical path requests pretrained DINO weights through timm; when network and compatible local Hugging Face cache resolution both fail, the exception path may continue with randomly initialized encoder weights instead of failing closed.
- Referenced raw validation artifacts are not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
