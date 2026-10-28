"""Opt-in archive and real-runtime checks without installing PaintFE system-wide.

PAINTFE_PACKAGE_DIR must contain packages built by packaging/linux/build-packages.sh.
Run under Linux with python3 -m unittest tools.tests.test_linux_packages -v.
"""
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform.startswith('linux') and os.environ.get('PAINTFE_PACKAGE_DIR'),
                     'requires Linux and built DEB/RPM package fixtures')
class LinuxPackages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packages = Path(os.environ['PAINTFE_PACKAGE_DIR']).resolve()
        cls.work = tempfile.TemporaryDirectory(prefix='paintfe-package-review-')
        cls.root = Path(cls.work.name)
        subprocess.run(['dpkg-deb', '-x', str(cls.packages/'PaintFE-linux-x64.deb'), str(cls.root)], check=True)
        (cls.root/'usr/lib/paintfe/paintfe_storage.cfg').write_text('portable\n')
        shutil.copyfile(Path(__file__).resolve().parents[2]/'tests/fixtures/heif/RGBA_8__29x100.heif.png', cls.root/'input.png')

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_both_launchers_resolve_and_run_the_actual_editor_cli(self):
        for name in ('paintfe', 'PaintFE'):
            launcher = self.root/'usr/bin'/name
            self.assertEqual(os.readlink(launcher), '../lib/paintfe/PaintFE')
            self.assertEqual(launcher.resolve(), self.root/'usr/lib/paintfe/PaintFE')
            output = self.root/(name+'-output.png')
            result = subprocess.run([str(launcher), '--input', str(self.root/'input.png'), '--output', str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        desktop = (self.root/'usr/share/applications/PaintFE.desktop').read_text()
        self.assertIn('Exec=paintfe %F', desktop)

    def test_private_host_is_preserved_and_its_actual_self_test_runs(self):
        folder = self.root/'usr/lib/paintfe/paintdotnet-host'
        host = folder/'PaintFE.PaintDotNetHost'
        self.assertTrue(list(folder.glob('*.dll')))
        self.assertTrue(list(folder.glob('*.so')))
        result = subprocess.run([str(host), '--self-test'], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('self-test passed', result.stdout)

    def test_rpm_payload_contains_both_commands_and_matching_host(self):
        result = subprocess.run(['rpm', '-qpl', str(self.packages/'PaintFE-linux-x64.rpm')], capture_output=True, text=True, check=True)
        for path in ('/usr/bin/paintfe', '/usr/bin/PaintFE', '/usr/lib/paintfe/paintdotnet-host/PaintFE.PaintDotNetHost'):
            self.assertIn(path, result.stdout.splitlines())
        result = subprocess.run(['rpm', '-qp', '--scripts', str(self.packages/'PaintFE-linux-x64.rpm')], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), '')


if __name__ == '__main__':
    unittest.main()
