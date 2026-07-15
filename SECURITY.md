# Security Policy

## Reporting status

BaoIAD does not currently publish a verified private vulnerability-reporting
address. Do **not** open a public Issue or pull request containing exploit
details, credentials, private data, or an unpatched vulnerability.

Activation of a private reporting channel and confirmation of an accountable
Security owner are tracked as the release-blocking approval
`APP-SECURITY-CHANNEL` in
[the external approval register](docs/release/external_approvals.json). Until
that approval is complete and this file contains the verified channel, the
repository must not claim that private security reporting is available and the
public release security gate remains closed.

Repository maintainers must not replace this notice with a placeholder email,
personal address, public Issue link, or unverified GitHub team.

## What to include once the private channel is active

Provide only through the approved private channel:

- the affected BaoIAD version or exact commit;
- the affected config, component, and dependency versions;
- impact and realistic attack prerequisites;
- minimal reproduction steps or a proof of concept;
- whether credentials, datasets, checkpoints, or user data may be exposed;
- suggested mitigations, if known.

Do not attach third-party datasets, private checkpoints, secrets, or internal
service details unless the approved Security owner explicitly requests a safe
transfer method.

## Scope

Security reports may cover BaoIAD-authored source code, release tooling,
dependency handling, checkpoint loading, path handling, and published
repository automation. Vulnerabilities in upstream services or third-party
packages should also be reported to their owners; BaoIAD may coordinate a
repository-side mitigation when its users are affected.

No response-time or remediation-time commitment is active until the Security
owner and private channel have passed their external preflight. Any future SLA
must be approved and stated here rather than inferred from Issue activity.

## Release handling

Security fixes use a private triage branch or other organization-approved
process, receive Security and technical review, and pass the same exact-commit
release gates as other changes. Published tags are immutable: maintainers must
issue a new patch release and must not force-push or move an existing tag.
