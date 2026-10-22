"""Pinned native AVIF decoder build; returns an environment for the Rust helper.

Build-time downloads only. Nothing here is called by the PaintFE application.
"""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.5.4'
NASM_SHA256 = '3ee4782247bcb874378d02f7eab4e294a84d3d15f3f6ee2de2f47a46aa7226e6'
SOURCE_SHA256 = '686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5'


def download(url, path, sha256):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.download')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Source checksum mismatch: {path.name}')


def prepare(target, original_env):
    env = original_env.copy()
    work = ROOT / 'target' / 'avif-native'
    work.mkdir(parents=True, exist_ok=True)
    tools = work / 'tools'
    python = tools / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(tools)], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'meson==1.12.1', 'ninja==1.13.2'], check=True)
    tool_bin = python.parent
    env['PATH'] = str(tool_bin) + os.pathsep + env.get('PATH', '')
    env['TEMP'] = env['TMP'] = str(work / 'tmp')
    Path(env['TEMP']).mkdir(exist_ok=True)
    if os.name == 'nt' and not shutil.which('nasm', path=env['PATH']):
        # Official portable assembler, not installed into the user's system.
        archive = work / 'nasm-2.16.03-win64.zip'
        download('https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/win64/nasm-2.16.03-win64.zip', archive, NASM_SHA256)
        destination = work / 'nasm'
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not (destination / name).resolve().is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe assembler archive')
            zipped.extractall(destination)
        env['PATH'] = str(destination / 'nasm-2.16.03') + os.pathsep + env['PATH']
    archive = work / f'dav1d-{VERSION}.tar.xz'
    download(f'https://downloads.videolan.org/pub/videolan/dav1d/{VERSION}/{archive.name}', archive, SOURCE_SHA256)
    source = work / f'dav1d-{VERSION}'
    if not source.exists():
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if not (work / member.name).resolve().is_relative_to(work.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe decoder archive')
            tar.extractall(work, filter='data')
    build = work / (target or f'{platform.system()}-{platform.machine()}')
    meson = [str(python), '-m', 'mesonbuild.mesonmain']
    if not (build / 'build.ninja').exists():
        options = ['--buildtype=release', '-Ddefault_library=static', '-Denable_tools=false', '-Denable_tests=false']
        if os.name == 'nt': options += ['--vsenv', '-Db_vscrt=mt']
        if target == 'x86_64-apple-darwin' and platform.machine() == 'arm64':
            cross = work / 'macos-intel.ini'
            cross.write_text("[binaries]\nc = 'clang'\nar = 'ar'\nstrip = 'strip'\n[host_machine]\nsystem = 'darwin'\ncpu_family = 'x86_64'\ncpu = 'x86_64'\nendian = 'little'\n[built-in options]\nc_args = ['-arch', 'x86_64']\nc_link_args = ['-arch', 'x86_64']\n", encoding='utf-8')
            options += ['--cross-file', str(cross)]
        subprocess.run(meson + ['setup', str(build), str(source)] + options, env=env, check=True)
    subprocess.run(meson + ['compile', '-C', str(build), '-j', '2'], env=env, check=True)
    if os.name == 'nt': shutil.copy2(build / 'src/libdav1d.a', build / 'src/dav1d.lib')
    env.update(SYSTEM_DEPS_DAV1D_NO_PKG_CONFIG='1', SYSTEM_DEPS_DAV1D_LIB='dav1d',
               SYSTEM_DEPS_DAV1D_SEARCH_NATIVE=str(build / 'src'), SYSTEM_DEPS_DAV1D_LINK='static')
    return env, source / 'COPYING'
