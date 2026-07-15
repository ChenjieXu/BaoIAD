# Contributing to BaoIAD

Thank you for improving BaoIAD. Contributions should be reviewable,
reproducible, and honest about their validation boundary.

## Before opening an issue

- Use the repository Issue forms for bugs, feature requests, and documentation
  problems.
- Do not report a suspected vulnerability in a public Issue or pull request.
  Read [SECURITY.md](SECURITY.md) before sharing any security-sensitive detail.
- Dataset access, third-party services, and unpublished checkpoints are outside
  BaoIAD's support commitment unless the repository explicitly says otherwise.

## Development workflow

1. Fork the organization repository and branch from the current `master`.
2. Keep the change focused. Do not commit datasets, pretrained weights,
   experiment outputs, caches, credentials, or internal paths.
3. Follow the detailed
   [English contribution guide](docs/en/notes/contributing.md) for method,
   config, registry, test, and documentation conventions.
4. Add or update tests for changed behavior. Network, slow, optional, and GPU
   tests must remain separate from the required offline CPU gate.
5. Update provenance, licensing notes, and method-status records when copying,
   adapting, or depending on third-party work.
6. Open a pull request using the repository template and report exactly what
   was and was not validated.

## Local checks

Run the checks that apply to your change:

```bash
ruff check baoiad tests tools
python tools/check_method_inventory.py
pytest -m "not network and not gpu and not slow"
```

Documentation changes should also build both languages with warnings treated
as errors. Use the committed CI constraints when reproducing the Python 3.10
and 3.12 release jobs.

Passing CPU tests does not validate CUDA training, compiled CUDA operators,
peak GPU memory, or end-to-end GPU inference. If a change was not tested on a
real GPU, write **GPU not validated** in the pull request and do not imply GPU
support was verified.
