# WAIC release go/no-go

## Current repository decision: NO-GO

This document is the public checklist, not the signed decision record. The
repository currently contains unresolved release blockers and therefore does
not authorize a tag, GitHub Release, ReadTheDocs promotion, Zenodo version, or
WAIC publication.

The current NO-GO remains in force while any of the following is true:

- an item in the [external approval register](external_approvals.json) is not
  approved or remains release-blocking;
- the private security-reporting channel and accountable Security owner have
  not passed their receive-and-escalation test;
- the exact merged candidate has not produced final evidence for every
  required CI context and clean-install gate;
- Legal/OSS, Brand, Technical, Security, and Release owner decisions or
  production-service permission preflights are absent;
- launch support owner and backup assignments have not been recorded in the
  restricted decision evidence;
- GPU validation is in release scope but real-CUDA evidence is missing, or GPU
  validation is excluded without an approved scope limitation.

Repository text must not replace any of these items with assumed approval,
placeholder contacts, or an untested production account.

## Bind checks to the exact candidate

Run the gate from the exact merged commit intended for publication. Capture
the resulting SHA and command outputs in restricted release evidence; do not
write the SHA into this file because that edit would create a different
candidate.

```bash
test -z "$(git status --porcelain=v2)"
CANDIDATE_SHA="$(git rev-parse HEAD)"
test "$(git rev-parse "${CANDIDATE_SHA}^{commit}")" = "${CANDIDATE_SHA}"
git show --no-patch --format=fuller "${CANDIDATE_SHA}"

python tools/check_public_release.py
python tools/check_release_candidate.py --static-only
python tools/check_release_compliance.py
python tools/check_method_inventory.py
```

The restricted evidence record must bind the same `CANDIDATE_SHA` to:

- the exact diff and literal allowlist result;
- the `lint`, `release-policy`, `core-offline (3.10)`,
  `core-offline (3.12)`, `docs-en`, and `docs-zh` status contexts;
- Python 3.10/3.12 clean-install and package results;
- provenance, licensing, asset, company-identity, security, and community
  governance approvals;
- Technical and Brand review of the rendered English and Chinese public
  documentation;
- GitHub tag/Release, ReadTheDocs, and Zenodo permission preflights;
- the approved GPU scope decision and, when required, its evidence artifact;
- the launch support owner and backup.

After all recorded blockers are resolved, rerun the complete candidate gate on
that same clean commit:

```bash
python tools/check_release_candidate.py
python tools/check_release_compliance.py --release-gate
```

Any change after those commands creates a new candidate and requires the full
gate again.

## CPU and GPU are independent decisions

Green CPU checks prove only their stated CPU, static, package, offline, and
documentation scope. They do not prove CUDA availability, CUDA operator
execution, GPU training or inference, peak VRAM, or 37-method GPU execution.

If GPU validation is required for the approved release scope, run the manual
real-GPU workflow and require evidence for the exact candidate:

```bash
python tools/check_gpu_evidence.py \
  --repo-root . \
  --evidence /path/to/gpu-evidence.json \
  --require-validated
```

Missing, synthetic, stale, modified, or explicit `not_validated` evidence is a
NO-GO. If the owners instead exclude GPU validation from the public scope,
the signed decision must preserve **GPU not validated** in release notes and
must prohibit GPU-validation claims.

## Decision record

The restricted decision record must contain one explicit GO or NO-GO from
each required role:

| Role | Decision responsibility |
|---|---|
| Technical maintainer | Candidate correctness, compatibility, test evidence, and claim boundaries |
| Legal/OSS owner | License, provenance, redistribution, and third-party disposition |
| Brand owner | Company identity, assets, README, documentation, and WAIC wording |
| Security owner | Private reporting channel, security triage readiness, and security risk |
| Release owner | Exact-commit evidence, final decision, production execution, and rollback readiness |
| Launch support owner and backup | Publication monitoring and incident handoff readiness |

Public files contain role names only. Signatures, personal details, private
channels, legal records, and opaque approval references remain in restricted
evidence.

## Production action boundary

Only after every required role records GO may approved owners create the
annotated tag and GitHub Release, promote ReadTheDocs stable/latest, create the
matching Zenodo software version, or announce the WAIC release. If any item is
yellow, red, unavailable, or ambiguous, take no production action and prepare
a new exact candidate after remediation.
