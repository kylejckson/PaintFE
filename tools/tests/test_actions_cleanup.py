"""Safety checks for the read-only cleanup inventory."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cleanup', Path(__file__).resolve().parents[1]/'plan_actions_cleanup.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def artifact(id=1, name='PaintFE-appimage', date='2026-10-01T00:00:00Z'):
    return dict(id=id, name=name, created_at=date, size_in_bytes=100, workflow_run={'id': 10})


def run(**extra):
    return dict({'id':10, 'path':'.github/workflows/release.yml', 'status':'completed',
                 'conclusion':'success', 'head_branch':'v1.4.10'}, **extra)


def release(**extra):
    return dict({'tag_name':'v1.4.10', 'draft':False, 'published_at':'2026-10-02T00:00:00Z',
                 'assets':[dict(name=n, size=100, state='uploaded') for n in ('PaintFE-x86_64.AppImage', 'checksums-SHA256.txt')]}, **extra)


class CleanupTests(unittest.TestCase):
    def eligible(self, item=None, workflow=None, published=None):
        return cleanup.plan([item or artifact()], {10:workflow or run()},
                            [published or release()], NOW)[0]['candidate']

    def test_published_complete_successful_old_release_is_candidate(self):
        self.assertTrue(self.eligible())

    def test_active_and_unknown_runs_are_preserved(self):
        self.assertFalse(self.eligible(workflow=run(status='in_progress')))
        self.assertFalse(cleanup.plan([artifact()], {}, [release()], NOW)[0]['candidate'])

    def test_failed_cancelled_and_unpublished_releases_are_preserved(self):
        for conclusion in ('failure', 'cancelled'):
            with self.subTest(conclusion=conclusion):
                self.assertFalse(self.eligible(workflow=run(conclusion=conclusion)))
        self.assertFalse(self.eligible(published=release(draft=True)))
        self.assertFalse(self.eligible(published=release(published_at=None)))
        self.assertFalse(cleanup.plan([artifact()], {10:run()}, [], NOW)[0]['candidate'])

    def test_missing_incomplete_or_empty_release_assets_are_preserved(self):
        for assets in ([], [dict(name='PaintFE-x86_64.AppImage', size=0, state='uploaded')],
                       [dict(name='checksums-SHA256.txt', size=100, state='uploaded')]):
            with self.subTest(assets=assets):
                self.assertFalse(self.eligible(published=release(assets=assets)))

    def test_recent_release_and_unknown_artifact_are_preserved(self):
        self.assertFalse(self.eligible(item=artifact(date='2026-10-21T12:00:00Z')))
        self.assertFalse(self.eligible(item=artifact(name='SARIF file')))

    def test_newest_portable_and_msix_are_always_preserved(self):
        for name in ('PaintFE-windows-x64', 'PaintFE-msix-store-submission'):
            with self.subTest(name=name):
                self.assertFalse(self.eligible(item=artifact(name=name)))

    def test_old_portable_is_candidate_but_recent_or_unverified_is_not(self):
        old=artifact(name='PaintFE-windows-x64')
        newer=artifact(id=2, name=old['name'], date='2026-10-21T00:00:00Z')
        workflows={10:run(path='.github/workflows/windows-portable.yml')}
        rows=cleanup.plan([old,newer],workflows,[],NOW)
        self.assertEqual([r['candidate'] for r in rows],[True,False])
        self.assertFalse(cleanup.plan([old,newer],{10:run()},[],NOW)[0]['candidate'])

    def test_msix_within_submission_window_is_preserved(self):
        for date, expected in [('2026-10-01T00:00:00Z', True), ('2026-10-15T00:00:00Z', False)]:
            old=artifact(name='PaintFE-msix-store-submission', date=date)
            new=artifact(id=2, name=old['name'], date='2026-10-21T00:00:00Z')
            self.assertEqual(cleanup.plan([old,new],{10:run()},[],NOW)[0]['candidate'], expected)

    def test_packs_need_both_platform_specific_published_files(self):
        item=artifact(name='optional-packs-build-macos-arm')
        assets=[dict(name=n,size=100,state='uploaded') for n in
                ('PaintFE-pack-cjk-macos-aarch64-v1.4.10.pfepack',
                 'PaintFE-pack-jxl-macos-aarch64-v1.4.10.pfepack', 'checksums-SHA256.txt')]
        self.assertTrue(self.eligible(item=item,published=release(assets=assets)))
        self.assertFalse(self.eligible(item=item,published=release(assets=assets[1:])))


if __name__ == '__main__':
    unittest.main()
