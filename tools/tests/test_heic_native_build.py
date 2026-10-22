"""Offline source rebuild and cross-platform native configuration regressions."""
import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'build_heic_native.py'
spec = importlib.util.spec_from_file_location('heic_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class HeicNativeBuildTests(unittest.TestCase):
    def test_included_sources_rebuild_without_download_and_include_actual_patch(self):
        parent = SCRIPT.parents[1] / 'target/heic-build-tests'
        parent.mkdir(parents=True, exist_ok=True)
        for target in (None, 'x86_64-apple-darwin', 'aarch64-apple-darwin', 'x86_64-unknown-linux-gnu'):
            with self.subTest(target=target), tempfile.TemporaryDirectory(dir=parent) as folder:
                root = Path(folder)
                plugin = root / 'libheif-1.23.6/libheif/plugins/encoder_kvazaar.cc'
                plugin.parent.mkdir(parents=True)
                plugin.write_text('api->config_init(config);\nconfig->lossless = encoder->lossless ? 1 : 0;')
                (root / 'libheif-1.23.6/COPYING').write_text('LGPL and GPL texts')
                (root / 'kvazaar-2.3.2').mkdir()
                (root / 'kvazaar-2.3.2/LICENSE').write_text('BSD text')
                (root / 'heic-encoder').mkdir()
                (root / 'heic-encoder/BUILDING.txt').write_text('build instructions')
                machine = 'arm64' if target and 'apple' in target else 'AMD64'
                with patch.object(builder, 'cmake_path', return_value='cmake'), patch.object(builder.platform, 'system', return_value='Windows'), patch.object(builder.platform, 'machine', return_value=machine), patch.object(builder.subprocess, 'run') as run, patch.object(builder.urllib.request, 'urlopen', side_effect=AssertionError('offline rebuild downloaded')):
                    binary, license_path, source_path, instructions = builder.prepare(target, {'PATH':'tools'}, root)
                self.assertEqual(run.call_count, 5)
                calls = [call.args[0] for call in run.call_args_list]
                self.assertIn('-DBUILD_SHARED_LIBS=OFF', calls[0])
                self.assertIn('-DBUILD_TESTS=OFF', calls[0])
                self.assertIn('-DUSE_CRYPTO=OFF', calls[0])
                if target and 'apple' in target:
                    arch = 'x86_64' if target.startswith('x86_64') else 'arm64'
                    self.assertIn(f'-DCMAKE_OSX_ARCHITECTURES={arch}', calls[0])
                    self.assertIn(f'-DCMAKE_OSX_ARCHITECTURES={arch}', calls[3])
                    if arch == 'x86_64':
                        self.assertIn('-DCMAKE_SYSTEM_PROCESSOR=x86_64', calls[0])
                        self.assertIn('-DCMAKE_SYSTEM_NAME=Darwin', calls[3])
                self.assertIn('config->threads = 4', plugin.read_text())
                self.assertIn('"preset", "fast"', plugin.read_text())
                self.assertIn('input_class == heif_image_input_class_alpha', plugin.read_text())
                self.assertIn('GPL-3.0-or-later', license_path.read_text())
                self.assertTrue(instructions.is_file())
                with tarfile.open(source_path) as tar:
                    self.assertIn('build_heic_native.py', tar.getnames())
                    patched = tar.extractfile('libheif-1.23.6/libheif/plugins/encoder_kvazaar.cc').read()
                    self.assertIn(b'config->threads = 4', patched)
                self.assertIn('paintfe-heic-encoder', binary.name)

    def test_offline_rebuild_does_not_bootstrap_cmake_from_the_network(self):
        with patch.object(builder.shutil, 'which', return_value=None), patch.object(builder.os, 'name', 'posix'), patch.object(builder.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'CMake 3.25'):
                builder.cmake_path(Path('target'), {}, offline=True)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
