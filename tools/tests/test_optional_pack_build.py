"""Packager regression checks. Native compilation is mocked; signing/ZIP are real."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / 'build_optional_packs.py'
spec = importlib.util.spec_from_file_location('pack_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PackBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_root = SCRIPT.parents[1] / 'target' / 'pack-builder-tests'
        cls.test_root.mkdir(parents=True, exist_ok=True)

    def test_old_python_has_actionable_error_before_importing_dependencies(self):
        with patch.object(sys, 'version_info', (3, 10, 12)):
            with self.assertRaisesRegex(SystemExit, 'Python 3.11 or later'):
                runpy.run_path(str(SCRIPT), run_name='packager_version_check')

    def test_platform_mapping(self):
        for system, machine, expected in [
            ('Windows', 'AMD64', ('windows', 'x86_64')),
            ('Linux', 'x86_64', ('linux', 'x86_64')),
            ('Darwin', 'arm64', ('macos', 'aarch64')),
            ('Darwin', 'x86_64', ('macos', 'x86_64')),
        ]:
            with self.subTest(system=system, machine=machine):
                with patch.object(builder.platform, 'system', return_value=system), patch.object(builder.platform, 'machine', return_value=machine):
                    self.assertEqual(builder.platform_id(), expected)
        self.assertEqual(builder.platform_id('x86_64-apple-darwin'), ('macos', 'x86_64'))

    def test_signed_packs_and_ci_environment_match_every_release_platform(self):
        for system, machine, target, expected_platform in [
            ('Windows', 'AMD64', None, 'windows-x86_64'),
            ('Linux', 'x86_64', None, 'linux-x86_64'),
            ('Darwin', 'arm64', None, 'macos-aarch64'),
            ('Darwin', 'arm64', 'x86_64-apple-darwin', 'macos-x86_64'),
        ]:
            with self.subTest(system=system, target=target), tempfile.TemporaryDirectory(dir=self.test_root) as directory:
                root = Path(directory)
                (root/'assets/fonts').mkdir(parents=True)
                (root/'optional/jxl-encoder').mkdir(parents=True)
                (root/'Cargo.toml').write_text('[package]\nversion="1.2.3"\n')
                (root/'assets/fonts/NotoSansCJK-Regular.otf').write_bytes(b'font')
                (root/'assets/fonts/NotoSansCJK-OFL.txt').write_text('font licence')
                (root/'optional/jxl-encoder/LICENSE.txt').write_text('encoder licence')
                binary_root = root/'target'
                if target:
                    binary_root /= target
                binary_root /= 'release'
                binary_root.mkdir(parents=True)
                suffix = '.exe' if system == 'Windows' else ''
                (binary_root/f'paintfe-jxl-encoder{suffix}').write_bytes(b'helper')
                (binary_root/f'paintfe-avif-codec{suffix}').write_bytes(b'avif helper')
                (binary_root/f'paintfe-heic-encoder{suffix}').write_bytes(b'heic helper')
                (root/'sources.tar.gz').write_bytes(b'sources')
                (binary_root/f'PaintFE{suffix}').write_bytes(b'editor')
                env_file = root/'github-env'
                seed = '11'*32  # Public test fixture, never used for production.
                stale=root/'target/optional-packs'/expected_platform/'release'/f'PaintFE-pack-heic-{expected_platform}-v1.2.3.pfepack'
                stale.parent.mkdir(parents=True,exist_ok=True)
                stale.write_bytes(b'cached legacy HEIC pack')
                argv = ['packager', '--release', '--github-env']
                if target:
                    argv += ['--target', target]
                with patch.object(builder, 'ROOT', root), patch.object(sys, 'argv', argv), patch.dict(os.environ, {'CI':'true', 'PAINTFE_PACK_SIGNING_KEY':seed, 'GITHUB_ENV':str(env_file)}, clear=True), patch.object(builder.platform, 'system', return_value=system), patch.object(builder.platform, 'machine', return_value=machine), patch.object(builder.build_avif_native, 'prepare', return_value=({}, root/'optional/jxl-encoder/LICENSE.txt')), patch.object(builder.subprocess, 'check_output', return_value=b'{"packages": []}'), patch.object(builder.subprocess, 'run') as native_build, contextlib.redirect_stdout(io.StringIO()):
                    builder.run()
                self.assertEqual(native_build.call_count, 3)
                for call in native_build.call_args_list:
                    self.assertNotIn('PAINTFE_PACK_SIGNING_KEY', call.kwargs['env'])
                exported = dict(line.split('=', 1) for line in env_file.read_text().splitlines())
                self.assertFalse(list((root/'target/optional-packs').rglob('*pack-heic*')))
                self.assertEqual(set(exported), {'PAINTFE_PACK_PUBLIC_KEY', 'PAINTFE_CJK_PACK_SIZE', 'PAINTFE_JXL_PACK_SIZE', 'PAINTFE_TEST_JXL_PACK', 'PAINTFE_TEST_AVIF_PACK', 'PAINTFE_AVIF_PACK_SIZE', 'PAINTFE_TEST_STORAGE_DIR'})
                jxl = Path(exported['PAINTFE_TEST_JXL_PACK'])
                self.assertTrue(jxl.is_file())
                self.assertEqual(jxl.parent.parent.name, expected_platform)
                public = builder.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed)).public_key()
                with zipfile.ZipFile(jxl) as archive:
                    envelope = json.loads(archive.read('manifest.json'))
                    public.verify(bytes.fromhex(envelope['signature']), envelope['manifest'].encode())
                    manifest = json.loads(envelope['manifest'])
                    self.assertEqual(manifest['platform'], expected_platform)
                    self.assertEqual(manifest['protocol'], 1)
                    self.assertEqual(archive.read(f'paintfe-jxl-encoder{suffix}'), b'helper')
                self.assertEqual(int(exported['PAINTFE_JXL_PACK_SIZE']), jxl.stat().st_size)
                fonts=list(jxl.parent.glob('*cjk-universal-*.pfepack'))
                self.assertEqual(len(fonts),1)
                with zipfile.ZipFile(fonts[0]) as archive:
                    signed=json.loads(archive.read('manifest.json'))
                    public.verify(bytes.fromhex(signed['signature']),signed['manifest'].encode())
                    self.assertEqual(json.loads(signed['manifest'])['platform'],'universal')
                self.assertNotIn(seed, env_file.read_text())

    def test_release_without_signing_key_stops_before_compilation(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as directory, patch.object(builder, 'ROOT', Path(directory)):
            (Path(directory)/'Cargo.toml').write_text('[package]\nversion="1.2.3"\n')
            with patch.object(sys, 'argv', ['packager', '--release']), patch.dict(os.environ, {'CI':'true'}, clear=True), patch.object(builder, 'platform_id', return_value=('windows', 'x86_64')), patch.object(builder.subprocess, 'run') as compile:
                with self.assertRaisesRegex(SystemExit, 'OPTIONAL_PACK_SIGNING_KEY'):
                    builder.run()
                compile.assert_not_called()


if __name__ == '__main__':
    unittest.main()
