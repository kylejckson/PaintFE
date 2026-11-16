"""Keep all build dependency downloads on the same complete hash-checked lock."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BuildRequirementsTests(unittest.TestCase):
    def locked_packages(self, filename):
        text = (ROOT / filename).read_text()
        blocks = re.split(r"\n\s*\n", text)
        packages = {}
        for block in blocks:
            lines = [line for line in block.splitlines() if line and not line.startswith("#")]
            if not lines:
                continue
            match = re.fullmatch(r"([a-z]+)==([0-9.]+) \\", lines[0])
            self.assertIsNotNone(match)
            hashes = re.findall(r"--hash=sha256:([a-f0-9]{64})", "\n".join(lines[1:]))
            self.assertGreater(len(hashes), 0)
            self.assertEqual(len(hashes), len(set(hashes)))
            packages[match[1]] = match[2]
        return packages

    def test_lock_contains_exact_versions_and_wheel_hashes_for_every_dependency(self):
        self.assertEqual(self.locked_packages(".github/requirements-build.txt"), {"cryptography": "46.0.7", "cffi": "2.0.0", "pycparser": "2.23"})
        self.assertEqual(self.locked_packages(".github/requirements-native.txt"), {"cmake": "3.31.6", "meson": "1.12.1", "ninja": "1.13.2"})

    def test_all_workflow_pip_installs_require_the_shared_hash_lock_and_wheels(self):
        installs = []
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            for line in path.read_text().splitlines():
                if re.search(r"\bpip(?:3)? install\b", line):
                    installs.append((path.name, line))
                    for option in ["--require-hashes", "--only-binary=:all:", "-r .github/requirements-build.txt"]:
                        self.assertIn(option, line, path.name)
        self.assertEqual(len(installs), 8)

    def test_release_dependency_install_steps_do_not_receive_the_signing_key(self):
        release = (ROOT / ".github/workflows/release.yml").read_text()
        for step in release.split("      - name:"):
            if "pip install" in step:
                self.assertNotIn("OPTIONAL_PACK_SIGNING_KEY", step)
                self.assertNotIn("PAINTFE_PACK_SIGNING_KEY", step)


if __name__ == "__main__":
    unittest.main()
