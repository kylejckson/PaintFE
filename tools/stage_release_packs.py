"""Collect and verify all signed release packs before publishing them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
from urllib.request import urlopen
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PLATFORMS = ('windows-x86_64', 'linux-x86_64', 'macos-aarch64', 'macos-x86_64')
MAX_FILE = 128 * 1024 * 1024


def expected_packs(tag):
    if not re.fullmatch(r'v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?', tag):
        raise ValueError('Expected a v-prefixed release version')
    return {f'PaintFE-pack-{pack}-{platform}-{tag}.pfepack': (pack, platform)
            for platform in PLATFORMS for pack in ('cjk', 'jxl', 'avif', 'heic')}


def envelope(path):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('manifest.json')
        if info.file_size > 64 * 1024:
            raise ValueError('Oversized pack manifest')
        return json.loads(archive.read(info))


def embedded_key(core, signed):
    """Find the actual verification key compiled into the matching core EXE."""
    keys = set()
    signature = bytes.fromhex(signed['signature'])
    message = signed['manifest'].encode('utf-8')
    for match in re.finditer(rb'[0-9a-fA-F]{64,}', Path(core).read_bytes()):
        value = match.group()
        for offset in range(len(value) - 63):
            candidate = bytes.fromhex(value[offset:offset + 64].decode('ascii'))
            try:
                Ed25519PublicKey.from_public_bytes(candidate).verify(signature, message)
            except InvalidSignature:
                continue
            keys.add(candidate)
    if len(keys) != 1:
        raise ValueError('Pack does not match a unique signing key in the core EXE')
    return Ed25519PublicKey.from_public_bytes(keys.pop())


def verify_pack(path, tag, pack, platform, key):
    signed = envelope(path)
    key.verify(bytes.fromhex(signed['signature']), signed['manifest'].encode('utf-8'))
    manifest = json.loads(signed['manifest'])
    identity = (manifest['schema'], manifest['protocol'], manifest['app_version'],
                manifest['pack'], manifest['platform'])
    if identity != (1, 1, tag[1:], pack, platform):
        raise ValueError(f'Wrong pack identity: {path.name}')
    payload = ('NotoSansCJK-Regular.otf' if pack == 'cjk' else
               f'paintfe-{"avif-codec" if pack == "avif" else "heic-encoder" if pack == "heic" else "jxl-encoder"}' + ('.exe' if platform.startswith('windows-') else ''))
    names = {payload, 'LICENSE.txt'}
    if pack == 'heic': names |= {'SOURCES.tar.gz', 'BUILDING.txt'}
    if Counter(f['name'] for f in manifest['files']) != Counter(names):
        raise ValueError(f'Invalid manifest file list: {path.name}')
    with zipfile.ZipFile(path) as archive:
        if Counter(archive.namelist()) != Counter(names | {'manifest.json'}):
            raise ValueError(f'Unexpected or duplicate archive files: {path.name}')
        for file in manifest['files']:
            info = archive.getinfo(file['name'])
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in (0, stat.S_IFREG) or not 0 < info.file_size <= MAX_FILE:
                raise ValueError(f'Invalid archive entry: {path.name}')
            data = archive.read(info)
            if len(data) != file['size'] or hashlib.sha256(data).hexdigest() != file['sha256']:
                raise ValueError(f'Pack payload integrity failed: {path.name}')


def stage(source, destination, tag, core):
    expected = expected_packs(tag)
    found = list(Path(source).rglob('*.pfepack'))
    if Counter(p.name for p in found) != Counter(expected.keys()):
        raise ValueError('Expected exactly sixteen unique release packs for this version')
    paths = {p.name: p for p in found}
    reference = paths[f'PaintFE-pack-jxl-windows-x86_64-{tag}.pfepack']
    key = embedded_key(core, envelope(reference))
    for name, (pack, platform) in expected.items():
        verify_pack(paths[name], tag, pack, platform, key)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Pack staging directory must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected):
        shutil.copyfile(paths[name], destination / name)
    return sorted(expected)


def verify_published(directory, tag):
    """Check every public download URL and its bytes, not just release metadata."""
    for name in sorted(expected_packs(tag)):
        path = Path(directory) / name
        expected = hashlib.sha256(path.read_bytes()).digest()
        digest = hashlib.sha256()
        url = f'https://github.com/kylejckson/PaintFE/releases/download/{tag}/{name}'
        with urlopen(url, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
        if digest.digest() != expected:
            raise ValueError(f'Published pack hash mismatch: {name}')
        print(f'Public download verified: {name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--source')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--core')
    parser.add_argument('--verify-published', action='store_true')
    args = parser.parse_args()
    if args.verify_published:
        verify_published(args.destination, args.tag)
    else:
        if not args.source or not args.core:
            parser.error('Staging requires --source and --core')
        for name in stage(args.source, args.destination, args.tag, args.core):
            print(f'Verified and staged: {name}')


if __name__ == '__main__':
    main()
