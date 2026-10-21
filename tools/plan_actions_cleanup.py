"""Read-only Actions artifact cleanup inventory. Never deletes anything.

Requires an authenticated gh CLI. Review the JSON before approving deletions.
"""
import argparse
from datetime import datetime, timezone
import json
import re
import subprocess


RELEASE_ASSETS = {
    'PaintFE-windows': {'PaintFE.exe', 'PaintFE-windows-x64.zip'},
    'PaintFE-linux': {'PaintFE', 'PaintFE-linux-x64.tar.gz'},
    'PaintFE-appimage': {'PaintFE-x86_64.AppImage'},
    'PaintFE-macos-arm64': {'PaintFE-macOS-arm64.dmg'},
    'PaintFE-macos-x86_64': {'PaintFE-macOS-x86_64.dmg'},
    'PaintFE-linux-packages': {'PaintFE-linux-x64.deb', 'PaintFE-linux-x64.rpm'},
}
PACK_PLATFORMS = {
    'optional-packs-build-windows': 'windows-x86_64',
    'optional-packs-build-linux': 'linux-x86_64',
    'optional-packs-build-macos-arm': 'macos-aarch64',
    'optional-packs-build-macos-x86': 'macos-x86_64',
}


def pages(repo, endpoint):
    result = subprocess.run(['gh', 'api', '--paginate', '--slurp', f'repos/{repo}/{endpoint}'],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    return json.loads(result.stdout)


def plan(artifacts, runs, releases, now):
    published = {r['tag_name']: r for r in releases
                 if not r.get('draft') and r.get('published_at')}
    live = [a for a in artifacts if not a.get('expired')]
    latest = {}
    for name in ('PaintFE-windows-x64', 'PaintFE-msix-store-submission'):
        matching = [a for a in live if a['name'] == name]
        if matching:
            latest[name] = max(matching, key=lambda a: a['created_at'])['id']
    rows = []
    for artifact in live:
        name = artifact['name']
        run = runs.get(artifact['workflow_run']['id'])
        age = (now - datetime.fromisoformat(artifact['created_at'].replace('Z', '+00:00'))).total_seconds() / 86400
        candidate, reason = False, 'Unrecognized artifact; preserve'
        if run is None or run.get('status') != 'completed':
            reason = 'Active or unverified workflow; preserve'
        elif latest.get(name) == artifact['id']:
            reason = 'Newest portable/MSIX artifact; preserve'
        elif name == 'PaintFE-windows-x64' and run.get('path') == '.github/workflows/windows-portable.yml':
            candidate = age >= 3
            reason = 'Older development build' if candidate else 'Development build younger than three days'
        elif name == 'PaintFE-msix-store-submission' and run.get('path') == '.github/workflows/release.yml':
            candidate = age >= 14
            reason = 'Older MSIX; confirm submission/download before deletion' if candidate else 'MSIX within fourteen-day submission window'
        elif (name in RELEASE_ASSETS or name in PACK_PLATFORMS) and run.get('path') == '.github/workflows/release.yml':
            release = published.get(run.get('head_branch'))
            if age < 1:
                reason = 'Release artifact younger than one day; preserve'
            elif run.get('conclusion') != 'success' or not release:
                reason = 'Release unsuccessful or unpublished; preserve'
            else:
                expected = RELEASE_ASSETS.get(name)
                if name in PACK_PLATFORMS:
                    platform = PACK_PLATFORMS[name]
                    expected = {f'PaintFE-pack-{pack}-{platform}-{release["tag_name"]}.pfepack' for pack in ('cjk', 'jxl')}
                available = {a['name'] for a in release.get('assets', []) if a.get('size', 0) > 0 and a.get('state') == 'uploaded'}
                candidate = (expected | {'checksums-SHA256.txt'}) <= available
                reason = 'Successful published release contains the final downloads and checksums' if candidate else 'Final release assets incomplete; preserve'
        rows.append(dict(id=artifact['id'], name=name, run_id=artifact['workflow_run']['id'],
                         bytes=artifact['size_in_bytes'], age_days=round(age, 2),
                         candidate=candidate, reason=reason))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='kylejckson/PaintFE')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
        parser.error('Expected owner/repository')
    artifacts = [a for p in pages(args.repo, 'actions/artifacts?per_page=100') for a in p['artifacts']]
    releases = [r for p in pages(args.repo, 'releases?per_page=100') for r in p]
    runs = {}
    for workflow in ('release.yml', 'windows-portable.yml'):
        for page in pages(args.repo, f'actions/workflows/{workflow}/runs?per_page=100'):
            runs.update({r['id']: r for r in page['workflow_runs']})
    rows = plan(artifacts, runs, releases, datetime.now(timezone.utc))
    print(json.dumps(dict(dry_run=True, repository=args.repo, artifacts=rows,
                          current_bytes=sum(a['bytes'] for a in rows),
                          candidate_bytes=sum(a['bytes'] for a in rows if a['candidate']),
                          candidate_count=sum(a['candidate'] for a in rows)), indent=2))


if __name__ == '__main__':
    main()
