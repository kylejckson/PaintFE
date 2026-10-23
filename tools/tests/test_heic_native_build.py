"""Offline source rebuild and cross-platform native configuration regressions."""
import importlib.util
import json
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
    @classmethod
    def setUpClass(cls):
        (SCRIPT.parents[1] / 'target/heic-build-tests').mkdir(parents=True, exist_ok=True)

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
                host = 'Darwin' if target and 'apple' in target else 'Linux' if target else 'Windows'
                machine = 'arm64' if host == 'Darwin' else 'AMD64'
                toolchain = dict(generator='Visual Studio 18 2026', instance='C:/VS/2026',
                                 minimum=(4, 2), cache='vs18-fixture')
                with patch.object(builder, 'windows_toolchain', return_value=toolchain), patch.object(builder, 'cmake_path', return_value='cmake'), patch.object(builder.platform, 'system', return_value=host), patch.object(builder.platform, 'machine', return_value=machine), patch.object(builder.subprocess, 'run') as run, patch.object(builder.urllib.request, 'urlopen', side_effect=AssertionError('offline rebuild downloaded')):
                    binary, license_path, source_path, instructions = builder.prepare(target, {'PATH':'tools'}, root)
                self.assertEqual(run.call_count, 5)
                calls = [call.args[0] for call in run.call_args_list]
                self.assertIn('-DBUILD_SHARED_LIBS=OFF', calls[0])
                self.assertIn('-DBUILD_TESTS=OFF', calls[0])
                self.assertIn('-DUSE_CRYPTO=OFF', calls[0])
                if host == 'Windows':
                    for command in (calls[0], calls[3]):
                        self.assertIn('Visual Studio 18 2026', command)
                        self.assertIn('-DCMAKE_GENERATOR_INSTANCE=C:/VS/2026', command)
                    self.assertIn('vs18-fixture', str(binary))
                else:
                    self.assertNotIn('-DCMAKE_GENERATOR_INSTANCE=C:/VS/2026', calls[0])
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

    def discover(self, instances):
        with patch.object(builder.shutil, 'which', return_value='vswhere.exe'), patch.object(builder.subprocess, 'check_output', return_value=json.dumps(instances)) as query:
            toolchain = builder.windows_toolchain({'PATH': 'tools'})
        command = query.call_args.args[0]
        self.assertIn('-products', command)
        self.assertIn('*', command)
        self.assertIn('Microsoft.VisualStudio.Component.VC.Tools.x86.x64', command)
        self.assertEqual(query.call_args.kwargs['encoding'], 'utf-8-sig')
        return toolchain

    def test_vs2022_uses_existing_generator_and_cmake_requirement(self):
        toolchain = self.discover([dict(installationVersion='17.14.1', installationPath='C:/VS/2022')])
        self.assertEqual(toolchain['generator'], 'Visual Studio 17 2022')
        self.assertEqual(toolchain['minimum'], (3, 25))

    def test_vs2026_only_runner_uses_new_generator(self):
        toolchain = self.discover([dict(installationVersion='18.5.1', installationPath='C:/VS/18')])
        self.assertEqual(toolchain['generator'], 'Visual Studio 18 2026')
        self.assertEqual(toolchain['minimum'], (4, 2))

    def test_newest_supported_installation_and_separate_cache(self):
        old = dict(installationVersion='17.14.1', installationPath='C:/VS/2022')
        new = dict(installationVersion='18.5.1', installationPath='C:/VS/18')
        newest = self.discover([new, old])
        self.assertEqual(newest['instance'], new['installationPath'])
        self.assertNotEqual(newest['cache'], self.discover([old])['cache'])
        moved = dict(new, installationPath='C:/VS/Other')
        self.assertNotEqual(newest['cache'], self.discover([moved])['cache'])

    def test_missing_cpp_installation_has_actionable_error(self):
        for instances in ([], [dict(installationVersion='16.0', installationPath='C:/VS/2019')]):
            with self.subTest(instances=instances), self.assertRaisesRegex(RuntimeError, r'x64 C\+\+ tools'):
                self.discover(instances)

    def test_vs2026_rejects_old_cmake_without_network_in_offline_mode(self):
        toolchain = dict(instance='C:/VS/nonexistent', minimum=(4, 2))
        with patch.object(builder.shutil, 'which', return_value='cmake'), patch.object(builder.subprocess, 'check_output', return_value='cmake version 3.31.6'), patch.object(builder.subprocess, 'run') as install:
            with self.assertRaisesRegex(RuntimeError, 'CMake 4.2'):
                builder.cmake_path(Path('target'), {}, offline=True, toolchain=toolchain)
            install.assert_not_called()

    def test_vs2026_bootstraps_pinned_compatible_cmake(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[1] / 'target/heic-build-tests') as folder, patch.object(builder.shutil, 'which', return_value=None), patch.object(builder.subprocess, 'run') as install:
            path = builder.cmake_path(Path(folder), {}, toolchain=dict(instance='C:/VS/nonexistent', minimum=(4, 2)))
            self.assertIn('cmake-4.2.3', path)
            self.assertTrue(any('cmake==4.2.3' in call.args[0] for call in install.call_args_list))

    def test_compatible_cmake_does_not_install_anything(self):
        with patch.object(builder.shutil, 'which', return_value='cmake'), patch.object(builder.subprocess, 'check_output', return_value='cmake version 4.2.3'), patch.object(builder.subprocess, 'run') as install:
            self.assertEqual(builder.cmake_path(Path('target'), {}, toolchain=dict(instance='C:/VS/nonexistent', minimum=(4, 2))), 'cmake')
            install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
