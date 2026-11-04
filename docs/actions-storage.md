# Actions Storage and Release Artifacts

Published GitHub Releases keep the permanent application downloads and optional packs. Actions artifacts are temporary files used to build those releases.

## Artifact Retention Periods

| Artifact | Retention |
| --- | --- |
| Intermediate release builds and packs | 1 day |
| Microsoft Store submission MSIX | 14 days |
| Manual Windows portable development build | 3 days |

Windows portable development builds are created only through the `Actions > Windows Portable Build > Run` workflow. Normal pushes still run CI. Download Store submissions within 14 days. If release intermediates expire before the release is published or before a retry, rerun the build jobs that create them before retrying publication.

Changing these workflows does not shorten the expiry date of existing artifacts. Published release downloads are not affected.

## Release Staging

Before publishing a release, `tools/stage_release_packs.py` gathers nested pack artifacts into a single flat directory and requires exactly 8 correctly named packs for that release version. It verifies signatures against the public key embedded in the matching Windows core EXE, checks manifest identities and payload hashes, and rejects missing, duplicate, or unexpected packs. All supported platforms use the same release signing key. No executable is run during verification.

Checksum pipelines and unmatched release uploads fail the job. After publishing, the workflow downloads each of the 8 public pack URLs and compares its hash with the staged file. `tools/tests/test_stage_release_packs.py` covers nested artifact paths, missing and duplicate packs, version and platform mismatches, payload tampering, core-key mismatches, and public download failures.

## Reviewing Existing Artifacts

Run this read-only report with an authenticated GitHub CLI:

```powershell
python tools/plan_actions_cleanup.py --repo kylejckson/PaintFE
```

The script has no delete option. It keeps active or unverified runs, unsuccessful or unpublished releases, the newest portable build and MSIX, and unknown artifacts. Older successful release intermediates are only candidates when the expected final downloads and checksum file already exist on the release. This check looks at metadata, not file contents. Review candidates before any separate cleanup, and confirm that older MSIX files have already been downloaded or submitted. Never delete release assets as part of Actions cleanup.

## Free Plan and Billing

GitHub Free gives 500 MB of artifact storage shared with GitHub Packages. Actions caches have a separate 10 GB limit per repository. Standard runners for public repositories are free; check account billing before blaming an account-wide warning on this public repository.

Storage charges are applied hourly. Deleting an artifact reduces future storage use, but it does not remove storage already charged. These workflows do not change account budgets, billing settings, or cache limits.

Sources: [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions) and [removing workflow artifacts](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/remove-workflow-artifacts).
