import importlib.util
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("aur_release", ROOT / "tools/prepare_aur_release.py")
aur = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aur)
SHA = "a" * 40
MAIN = "b" * 40


class AurReleaseTests(unittest.TestCase):
    def test_api_decodes_utf8_metadata_including_non_ascii_release_text(self):
        original = subprocess.check_output
        def reply(_command, **options):
            return original([sys.executable, '-c',
                             'import sys; sys.stdout.buffer.write(b\'{"title":"\\xe2\\x80\\x9d"}\')'],
                            **options)
        with patch.object(aur.subprocess, 'check_output', side_effect=reply):
            self.assertEqual(aur.api('test-endpoint')['title'], '\u201d')

    def response(self, endpoint):
        if "/releases/" in endpoint:
            return {"tag_name": "v1.5.1", "draft": False, "published_at": "2026-11-16",
                    "assets": [{"name": n} for n in ["PaintFE-linux-x64.tar.gz", "checksums-SHA256.txt"]]}
        if "/commits/" in endpoint:
            return {"sha": SHA}
        if "/git/ref/" in endpoint:
            return {"object": {"sha": MAIN}}
        return {"status": "ahead"}

    def test_owner_release_and_manual_retry_are_accepted(self):
        self.assertEqual(aur.verified_version("v1.5.1", aur.REPOSITORY, "push", SHA, self.response), "1.5.1")
        self.assertEqual(aur.verified_version("1.5.1", aur.REPOSITORY, "workflow_dispatch", request=self.response), "1.5.1")

    def test_untrusted_events_repositories_and_shell_strings_are_rejected_before_network(self):
        def no_network(_):
            self.fail("Invalid input reached the API")
        for repo, event, version in [("fork/PaintFE", "push", "1.5.1"),
                                     (aur.REPOSITORY, "pull_request", "1.5.1"),
                                     (aur.REPOSITORY, "workflow_run", "1.5.1")]:
            with self.assertRaises(ValueError):
                aur.verified_version(version, repo, event, SHA, no_network)
        for version in ["v$(id)", "1.5.1\nkey=value", "1.5.1;id", "01.5.1", "1.5.1/other", "1.5.1-rc1", ""]:
            with self.assertRaises(ValueError):
                aur.verified_version(version, aur.REPOSITORY, "push", SHA, no_network)

    def test_drafts_missing_assets_foreign_tags_and_unmerged_commits_are_rejected(self):
        for field, value in [("draft", True), ("published_at", None), ("assets", []), ("tag_name", "v2.0.0")]:
            def request(endpoint):
                result = self.response(endpoint)
                if "/releases/" in endpoint:
                    result[field] = value
                return result
            with self.assertRaises(ValueError):
                aur.verified_version("1.5.1", aur.REPOSITORY, "push", SHA, request)
        for status in ["behind", "diverged", "unknown"]:
            with self.assertRaises(ValueError):
                aur.verified_version("1.5.1", aur.REPOSITORY, "push", SHA,
                                     lambda e: {"status": status} if "/compare/" in e else self.response(e))
        for sha in ["", "c" * 40]:
            with self.assertRaises(ValueError):
                aur.verified_version("1.5.1", aur.REPOSITORY, "push", sha, self.response)

    def test_preparation_stamps_both_trusted_recipes_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "packaging/aur"
            source.mkdir(parents=True)
            for name in aur.RECIPES:
                (source / name).write_text("pkgname=paintfe\npkgver=1.0.0\npkgrel=1\n")
            aur.prepare(root, "1.5.1")
            for name in aur.RECIPES:
                self.assertIn("pkgver=1.5.1\n", (root / "target/aur" / name).read_text())
                self.assertIn("pkgver=1.0.0\n", (source / name).read_text())

    def test_workflows_keep_the_trust_boundary_explicit(self):
        workflow = (ROOT / ".github/workflows/aur-publish.yml").read_text()
        self.assertNotIn("workflow_run", workflow)
        self.assertIn("workflow_call:", workflow)
        self.assertIn("ref: main", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("contents: read", workflow)
        self.assertNotIn('run: |', workflow)
        release = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertIn("needs: release", release)
        self.assertIn("uses: ./.github/workflows/aur-publish.yml", release)


if __name__ == "__main__":
    unittest.main()
