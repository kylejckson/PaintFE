"""AV1 build planning and source-boundary regressions, without native compilation."""
import importlib.util
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('native', Path(__file__).resolve().parents[1] / 'build_avif_native.py')
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeAvifTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(SPEC.origin).resolve().parents[1] / 'target/avif-native-plan-tests'
        cls.scratch.mkdir(parents=True, exist_ok=True)

    def plan(self, system, target=None):
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            root = Path(directory)
            source = root / 'target/avif-native' / f'aom-{native.AOM_VERSION}'
            source.mkdir(parents=True)
            (source / 'CMakeLists.txt').write_text('# fixture')
            build = source.parent / ('aom-' + (target or f'{native.platform.system()}-{native.platform.machine()}'))
            build.mkdir()
            with patch.object(native, 'ROOT', root), patch.object(native, 'os', types.SimpleNamespace(name=system)), patch.object(native, 'download') as fetch, patch.object(native.shutil, 'which', side_effect=lambda name, **_: 'cmake' if name == 'cmake' else None), patch.object(native.subprocess, 'run') as run:
                env = {'PATH': '', 'CARGO_BUILD_JOBS': '1'}
                native.prepare_aom(target, env)
                args = run.call_args_list[0].args[0]
                self.assertIn('-DCONFIG_AV1_DECODER=0', args)
                self.assertIn('-DCONFIG_AV1_HIGHBITDEPTH=0', args)
                self.assertIn('-DBUILD_SHARED_LIBS=OFF', args)
                self.assertEqual(fetch.call_args.args[2], native.AOM_SHA256)
                build = run.call_args_list[1]
                self.assertEqual(build.args[0][-2:], ['--parallel', '1'])
                if system == 'nt':
                    self.assertTrue(build.kwargs['env']['_CL_'].endswith('/MP1'))
                    self.assertIn('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded', args)
                    self.assertTrue(env['PAINTFE_AOM_LIB_DIR'].endswith('Release'))
                return args

    def test_windows_static_crt_and_both_compiler_budgets(self):
        self.plan('nt')

    def test_linux_encoder_only_build(self):
        args = self.plan('posix', 'x86_64-unknown-linux-gnu')
        self.assertFalse(any('OSX' in value for value in args))

    def test_intel_and_arm_macos_use_explicit_matching_architectures(self):
        for target, arch in [('x86_64-apple-darwin', 'x86_64'), ('aarch64-apple-darwin', 'arm64')]:
            args = self.plan('posix', target)
            self.assertIn(f'-DCMAKE_OSX_ARCHITECTURES={arch}', args)
            self.assertIn(f'-DAOM_TARGET_CPU={arch}', args)

    def test_source_archive_cannot_escape_or_create_symlinks(self):
        for member_name, kind in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
                root = Path(directory)
                work = root / 'target/avif-native'
                work.mkdir(parents=True)
                with tarfile.open(work / f'aom-{native.AOM_VERSION}.tar.gz', 'w:gz') as tar:
                    member = tarfile.TarInfo(member_name)
                    member.type = kind
                    member.linkname = '../escape'
                    tar.addfile(member)
                with patch.object(native, 'ROOT', root), patch.object(native, 'download'), patch.object(native.shutil, 'which', return_value='cmake'):
                    with self.assertRaisesRegex(ValueError, 'Unsafe AV1 encoder archive'):
                        native.prepare_aom(None, {'PATH': ''})
