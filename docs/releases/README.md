# Versions and Change Notes

Keep described version history here, but keep executable version values in
[package metadata](../../pyproject.toml) and the runtime version module.
The root [CHANGELOG.md](../../CHANGELOG.md) is the canonical chronological change
log. Dated notes explain impact, compatibility, migration, and validation.

| Date | Project Status | Notes |
| --- | --- | --- |
| 2026-11-16 | Unreleased changes on package 0.4.0 | [Safety, retrieval upgrades, rtt, and researcher installation](2026-11-16.md) |

Dependency versions are distinct from the EEG-RAG package version. Current
resolutions live in `uv.lock`; generated requirements are installation profiles.
Use `python scripts/install_researcher.py --check` to report what is actually
installed. A version table in a dated note is a snapshot, not a live inventory.
Do not call an unreleased change a published version without a release.
