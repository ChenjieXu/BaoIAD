# PatchCore implementation provenance and reproducibility notes

This public summary is derived from the [method status inventory](method_status.json). It records the available repository-local provenance and validation boundary; it is not an assertion that the cited paper results can be independently reproduced from a clean public clone.

## Release status

- **Validation status:** Historical evidence
- **Public validation evidence distributed:** No
- **Runtime state:** <code>not_assessed</code> — Runtime availability has not been fully assessed; no clean-clone blocker was established beyond the separately recorded evidence and license limitations.
- **Method family:** Feature-memory / density
- **Registry entry:** <code>PatchCore</code>
- **Detector module:** <code>baoiad.models.detectors.patchcore</code>

## Public references

- **Paper:** [https://arxiv.org/abs/2106.08265](https://arxiv.org/abs/2106.08265)
- **Source repository:** Not recorded in the release inventory.
- **Source revision:** Not recorded in the release inventory.
- **Config README:** [configs/patchcore/README.md](../../configs/patchcore/README.md)
- **Release records:** [method status](method_status.json) · [known exceptions](exceptions.json)

## Implementation differences

Repository-local implementation audited against an unpinned local ADer snapshot, with approximate coreset and MMEngine adapters.

## Limitations

- The implementation reference has no recorded public revision in the alignment page.
- Referenced raw validation artifacts are not distributed.

Inventory assessment: <code>2026-07-15</code> against <code>upstream/master@e93614a01204c441fc85511765879ae031a360bb</code>.
