# Contributing to BaoIAD

Thank you for improving BaoIAD. Contributions should be reviewable,
reproducible, and honest about their validation boundary.

## Before opening an issue

- Use the repository Issue forms for bugs, feature requests, and documentation
  problems.
- Do not report a suspected vulnerability in a public Issue. Read
  [SECURITY.md](SECURITY.md) first. The private reporting channel is a
  release-blocking external approval and must be active before a public
  release.
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
python tools/check_public_release.py
pytest -m "not network and not gpu and not slow"
```

Documentation changes should also build both languages with warnings treated
as errors. Use the committed CI constraints when reproducing the Python 3.10
and 3.12 release jobs.

Passing CPU tests does not validate CUDA training, compiled CUDA operators,
peak GPU memory, or end-to-end GPU inference. If no real-GPU evidence exists,
write **GPU not validated** in the pull request and do not make a GPU-support
claim.

## Review and release-sensitive changes

- At least one approving reviewer other than the author is required.
- Code and config changes require technical-maintainer review.
- Public identity, top-level documentation, and release media require the
  approved brand role.
- License, provenance, redistribution, and security-policy changes require the
  Legal/OSS or Security role recorded in the release approval checklist.
- Release metadata and tag preparation require the Release owner and the
  exact-commit go/no-go described in the
  [release process](docs/en/notes/release_process.md).

Approved GitHub users or teams have not yet been supplied for `CODEOWNERS`.
Review ownership is therefore documented by role and must not be represented
as automatically enforced until the organization configures those teams.
