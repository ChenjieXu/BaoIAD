## Summary

Describe the user-visible outcome and why the change is needed.

## Scope

- Change type: bug fix / feature / method or config / documentation / other
- Affected methods, configs, datasets, or public APIs:
- Compatibility or migration impact:

## Validation evidence

List exact commands and concise results. Do not report a check as passing if it
was skipped or could not run.

- [ ] Relevant targeted tests pass.
- [ ] Required offline CPU checks pass on the applicable Python versions.
- [ ] Method inventory checks pass when applicable.
- [ ] English and Chinese documentation build with warnings treated as errors
      when documentation changed.
- [ ] New or changed third-party code/assets have provenance and license notes.
- [ ] No dataset, checkpoint, experiment output, secret, or internal path was
      added.

### GPU validation

Select and explain exactly one state:

- [ ] GPU not required for this change.
- [ ] Real-GPU evidence is attached: device, driver/CUDA, PyTorch build,
      compiled-op status, training/inference smoke, and peak memory are listed.
- [ ] GPU not validated. No GPU-support claim is made by this pull request.

CPU-only checks never count as GPU validation.

## Documentation impact

- [ ] User-facing behavior and migration notes are documented.
- [ ] English/Chinese public invariants remain aligned where applicable.
- [ ] The changelog is updated, or this change has no user-visible impact.

## Known limitations

State remaining risks, unavailable hardware, or checks not run. Use **not
validated** rather than implying success without evidence.
