"""Select and verify Mesa's software Vulkan driver for Linux CI, without a display."""
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile


def prepare(env=None, driver_dir=Path('/usr/share/vulkan/icd.d'), machine=None):
    env = dict(os.environ if env is None else env)
    arch = (machine or platform.machine()).lower()
    arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(arch, arch)
    if arch not in ('x86_64', 'aarch64'):
        raise RuntimeError(f'Unsupported Linux CI architecture: {arch}')
    candidates = [driver_dir / f'lvp_icd.{arch}.json', driver_dir / 'lvp_icd.json']
    driver = next((path for path in candidates if path.is_file()), None)
    if driver is None:
        raise RuntimeError('Lavapipe was not found. Install mesa-vulkan-drivers and vulkan-tools.')
    runtime = Path(env.get('RUNNER_TEMP', tempfile.gettempdir())) / 'paintfe-vulkan'
    runtime.mkdir(parents=True, mode=0o700, exist_ok=True)
    runtime.chmod(0o700)
    exported = dict(VK_ICD_FILENAMES=str(driver), VK_DRIVER_FILES=str(driver),
                    XDG_RUNTIME_DIR=str(runtime), LP_NUM_THREADS='2')
    if any('\n' in value or '\r' in value for value in exported.values()):
        raise RuntimeError('Newlines are not allowed in Vulkan CI paths')
    env.update(exported)
    try:
        probe = subprocess.run(['vulkaninfo', '--summary'], env=env, text=True,
                               capture_output=True, check=True, timeout=30)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Lavapipe preflight failed:\n{error.stdout or ""}{error.stderr or ""}') from error
    if not re.search(r'lavapipe|llvmpipe', probe.stdout, re.IGNORECASE):
        raise RuntimeError('Vulkan is available, but the selected device is not Lavapipe/llvmpipe')
    # Export only after validation, so a failed preflight cannot leave partial settings.
    if env.get('GITHUB_ENV'):
        with Path(env['GITHUB_ENV']).open('a', encoding='utf-8') as output:
            for name, value in exported.items():
                output.write(f'{name}={value}\n')
    print(probe.stdout.strip())
    return exported


if __name__ == '__main__':
    if platform.system() != 'Linux':
        raise SystemExit('This preflight is for Linux CI only')
    prepare()
