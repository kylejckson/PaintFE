"""Linux CI software GPU selection, including failure before exporting settings."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'check_ci_vulkan.py'
spec = importlib.util.spec_from_file_location('ci_vulkan', SCRIPT)
vulkan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vulkan)
TEST_ROOT = SCRIPT.parents[1] / 'target/vulkan-preflight-tests'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class VulkanPreflightTests(unittest.TestCase):
    def test_architecture_specific_and_generic_drivers_export_only_safe_variables(self):
        for machine, filename in [('AMD64', 'lvp_icd.x86_64.json'), ('arm64', 'lvp_icd.aarch64.json'), ('x86_64', 'lvp_icd.json')]:
            with self.subTest(machine=machine), tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
                root = Path(folder)
                (root / filename).write_text('{}')
                env = dict(RUNNER_TEMP=folder, GITHUB_ENV=str(root / 'github.env'))
                result = subprocess.CompletedProcess([], 0, stdout='deviceName = llvmpipe (LLVM)\n', stderr='')
                with patch.object(vulkan.subprocess, 'run', return_value=result) as probe, contextlib.redirect_stdout(io.StringIO()):
                    exported = vulkan.prepare(env, root, machine)
                self.assertEqual(exported['VK_ICD_FILENAMES'], str(root / filename))
                self.assertEqual(exported['VK_DRIVER_FILES'], exported['VK_ICD_FILENAMES'])
                self.assertEqual(exported['LP_NUM_THREADS'], '2')
                self.assertTrue(Path(exported['XDG_RUNTIME_DIR']).is_dir())
                self.assertEqual(probe.call_args.args[0], ['vulkaninfo', '--summary'])
                self.assertEqual(probe.call_args.kwargs['timeout'], 30)
                self.assertTrue(probe.call_args.kwargs['check'])
                self.assertEqual(set(line.split('=', 1)[0] for line in (root / 'github.env').read_text().splitlines()), set(exported))
                self.assertNotIn('VK_ICD_FILENAMES', env)

    def test_missing_driver_fails_before_probe_or_export(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder, patch.object(vulkan.subprocess, 'run') as probe:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'mesa-vulkan-drivers'):
                vulkan.prepare(dict(GITHUB_ENV=str(root / 'github.env')), root, 'x86_64')
            self.assertFalse((root / 'github.env').exists())
            probe.assert_not_called()

    def test_failed_probe_and_wrong_gpu_do_not_export(self):
        cases = [subprocess.CalledProcessError(1, ['vulkaninfo']),
                 subprocess.CompletedProcess([], 0, stdout='deviceName = hardware GPU', stderr='')]
        for result in cases:
            with self.subTest(result=result), tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
                root = Path(folder)
                (root / 'lvp_icd.json').write_text('{}')
                options = dict(side_effect=result) if isinstance(result, Exception) else dict(return_value=result)
                with patch.object(vulkan.subprocess, 'run', **options), self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    vulkan.prepare(dict(RUNNER_TEMP=folder, GITHUB_ENV=str(root / 'github.env')), root, 'x86_64')
                self.assertFalse((root / 'github.env').exists())

    def test_workflow_installs_driver_and_checks_it_before_the_real_save_test(self):
        workflow = (SCRIPT.parents[1] / '.github/workflows/ci.yml').read_text()
        self.assertIn('mesa-vulkan-drivers vulkan-tools', workflow)
        linux = workflow[workflow.index('  build-unix:'):]
        self.assertLess(linux.index('python tools/check_ci_vulkan.py'), linux.index('cargo test -j1 --lib toolbar_save_'))


if __name__ == '__main__':
    unittest.main()
