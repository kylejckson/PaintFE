# Actions storage and release artifacts

Published GitHub Releases hold the permanent application downloads and optional
packs. Actions artifacts are temporary files used to assemble those releases.

The workflows use these retention periods for new uploads:

| Artifact | Retention |
| --- | --- |
| Intermediate release builds and packs | 1 day |
| Microsoft Store submission MSIX | 14 days |
| Manual Windows portable development build | 3 days |

Windows portable development builds run only through Actions > Windows Portable
Build > Run workflow. Normal pushes still run CI. Download Store submissions
within fourteen days. If release intermediates expire before publication or a
retry, rerun the build jobs that produce them before retrying publication.

Changing these workflows does not shorten existing artifacts' expiry dates.
Published release downloads are unaffected.

## Reviewing existing artifacts

Run this read-only report using an authenticated GitHub CLI:

```powershell
python tools/plan_actions_cleanup.py --repo kylejckson/PaintFE
```

The script has no deletion option. It preserves active or unverified runs,
unsuccessful or unpublished releases, the newest portable build and MSIX, and
unknown artifacts. Older successful release intermediates are candidates only
when the expected final downloads and checksum file exist on the release.
This checks metadata, not matching file contents. Review candidates before any
separate deletion, and confirm older MSIX files have been downloaded or submitted.
Never delete release assets as part of Actions cleanup.

## Free plan and billing

GitHub Free lists 500 MB of artifact storage shared with GitHub Packages.
Actions caches have a separate 10 GB allowance per repository. Standard runners
for public repositories are free; account billing should be checked before
attributing an account-wide warning to this public repository.

Storage charges accrue hourly. Deletion reduces future storage usage but does
not remove usage already accrued. These workflows do not change account budgets,
billing settings or cache limits.

Sources: [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
and [removing workflow artifacts](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/remove-workflow-artifacts).
