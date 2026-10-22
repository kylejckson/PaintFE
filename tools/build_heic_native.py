"""Pinned, offline-capable HEIC export helper build and corresponding-source pack."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ('libheif-1.23.6', 'https://github.com/strukturag/libheif/archive/refs/tags/v1.23.6.tar.gz', 'c9a00b0452f5c16d72ef71e5e40c0e90fbe2ad37148d27a1b2f9bb21546f5f03'),
    ('kvazaar-2.3.2', 'https://github.com/ultravideo/kvazaar/archive/refs/tags/v2.3.2.tar.gz', 'ddd0038696631ca5368d8e40efee36d2bbb805854b9b1dda8b12ea9b397ea951'),
)


def cmake_path(work, env, offline=False):
    found = shutil.which('cmake')
    candidates = [found] if found else []
    if os.name == 'nt':
        vs = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Microsoft Visual Studio/2022'
        candidates += sorted(vs.glob('*/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'))
    for candidate in candidates:
        version = subprocess.check_output([str(candidate), '--version'], env=env, text=True)
        match = re.search(r'version (\d+)\.(\d+)', version)
        if match and tuple(map(int, match.groups())) >= (3, 25):
            return str(candidate)
    if offline:
        raise RuntimeError('Install CMake 3.25 or later to rebuild the included HEIC source')
    tools = work / 'tools'
    python = tools / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(tools)], env=env, check=True)
    cmake = python.parent / ('cmake.exe' if os.name == 'nt' else 'cmake')
    if not cmake.exists():
        subprocess.run([str(python), '-m', 'pip', 'install', 'cmake==3.31.6'], env=env, check=True)
    return str(cmake)


def prepare(target=None, env=None, source_root=None):
    env = dict(os.environ if env is None else env)
    work = Path(source_root).resolve() if source_root else ROOT / 'target/heic-native'
    work.mkdir(parents=True, exist_ok=True)
    helper = work / 'heic-encoder' if source_root else ROOT / 'optional/heic-encoder'
    for name, url, digest in SOURCES:
        source = work / name
        if source_root:
            if not source.is_dir():
                raise ValueError(f'Missing included source: {name}')
            continue
        archive = work / f'{name}.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            temporary.replace(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError(f'HEIC source checksum mismatch: {name}')
        if not source.exists():
            with tarfile.open(archive) as tar:
                for item in tar.getmembers():
                    if not (work / item.name).resolve().is_relative_to(work.resolve()) or not (item.isfile() or item.isdir()):
                        raise ValueError('Unsafe HEIC source archive')
                tar.extractall(work, filter='data')
    # Kvazaar's libheif plugin does not expose preset/thread controls publicly.
    plugin = work / 'libheif-1.23.6/libheif/plugins/encoder_kvazaar.cc'
    text = plugin.read_text(encoding='utf-8')
    needle = 'api->config_init(config);'
    addition = '\n  api->config_parse(config, "preset", "fast"); // PaintFE\n  config->threads = 4; // PaintFE'
    if '// PaintFE' not in text:
        if text.count(needle) != 1:
            raise ValueError('Unexpected pinned Kvazaar integration source')
        plugin.write_text(text.replace(needle, needle + addition), encoding='utf-8')
    text = plugin.read_text(encoding='utf-8')
    lossy_alpha = 'config->lossless = encoder->lossless ? 1 : 0;'
    exact_alpha = 'config->lossless = (encoder->lossless || input_class == heif_image_input_class_alpha) ? 1 : 0; // PaintFE exact alpha'
    if lossy_alpha in text:
        plugin.write_text(text.replace(lossy_alpha, exact_alpha), encoding='utf-8')
    elif exact_alpha not in text:
        raise ValueError('Unexpected pinned HEIC alpha configuration')
    build = work / ('build-' + (target or f'{platform.system()}-{platform.machine()}'))
    build.mkdir(parents=True, exist_ok=True)
    prefix = build / 'install'
    cmake = cmake_path(work, env, offline=bool(source_root))
    common = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DCMAKE_POLICY_DEFAULT_CMP0091=NEW', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded']
    if os.name == 'nt':
        common += ['-G', 'Visual Studio 17 2022', '-A', 'x64']
    if target and 'apple' in target:
        common += [f'-DCMAKE_OSX_ARCHITECTURES={"x86_64" if target.startswith("x86_64") else "arm64"}']
        if target.startswith('x86_64') and platform.machine() in ('arm64', 'aarch64'):
            # Kvazaar selects x86 SIMD flags using CMAKE_SYSTEM_PROCESSOR.
            common += ['-DCMAKE_SYSTEM_NAME=Darwin', '-DCMAKE_SYSTEM_PROCESSOR=x86_64']
    def run(args):
        subprocess.run([cmake] + args, env=env, check=True)
    run(['-S', str(work / 'kvazaar-2.3.2'), '-B', str(build / 'kvazaar'), f'-DCMAKE_INSTALL_PREFIX={prefix}', '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTS=OFF', '-DBUILD_KVAZAAR_BINARY=OFF', '-DUSE_CRYPTO=OFF'] + common)
    run(['--build', str(build / 'kvazaar'), '--config', 'Release', '-j', '2'])
    run(['--install', str(build / 'kvazaar'), '--config', 'Release'])
    run(['-S', str(helper), '-B', str(build / 'helper'), f'-DHEIF_SOURCE={work / "libheif-1.23.6"}', f'-DCMAKE_PREFIX_PATH={prefix}'] + common)
    run(['--build', str(build / 'helper'), '--config', 'Release', '--target', 'paintfe-heic-encoder', '-j', '2'])
    binary = build / 'helper' / ('Release/paintfe-heic-encoder.exe' if os.name == 'nt' else 'paintfe-heic-encoder')
    notices = build / 'LICENSE.txt'
    notices.write_text('PaintFE HEIC helper: GPL-3.0-or-later. Complete corresponding source is in SOURCES.tar.gz.\n\n' + (work / 'libheif-1.23.6/COPYING').read_text(encoding='utf-8') + '\n\nKvazaar:\n' + (work / 'kvazaar-2.3.2/LICENSE').read_text(encoding='utf-8'), encoding='utf-8')
    sources = build / 'SOURCES.tar.gz'
    with tarfile.open(sources, 'w:gz') as tar:
        for name, _, _ in SOURCES:
            tar.add(work / name, arcname=name)
        tar.add(helper, arcname='heic-encoder')
        tar.add(Path(__file__), arcname='build_heic_native.py')
    return binary, notices, sources, helper / 'BUILDING.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target')
    parser.add_argument('--source-root')
    args = parser.parse_args()
    for path in prepare(args.target, source_root=args.source_root):
        print(path)
