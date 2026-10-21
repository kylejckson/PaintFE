"""Build release-bound signed packs and the core. Private keys exist in memory only.
Requires Python cryptography. Optional --target supports release cross-compilation.
"""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, zipfile
from pathlib import Path
if sys.version_info < (3, 11):
    raise SystemExit('Optional pack builds require Python 3.11 or later. Install Python 3.12 and retry.')
import tomllib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
ROOT = Path(__file__).resolve().parents[1]

def platform_id(target=None):
    os_name = ('windows' if 'windows' in target else 'macos' if 'apple' in target else 'linux') if target else {'Windows':'windows','Darwin':'macos','Linux':'linux'}[platform.system()]
    arch = target.split('-')[0] if target else {'AMD64':'x86_64','arm64':'aarch64'}.get(platform.machine(), platform.machine())
    return os_name, arch

def export_test_environment(output, pack_paths):
    metadata = json.loads((output/'build-env.json').read_text(encoding='utf-8'))
    metadata.update(PAINTFE_TEST_JXL_PACK=str(pack_paths['jxl']), PAINTFE_TEST_STORAGE_DIR=str(ROOT/'target'/'codec-test-storage'))
    with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as destination:
        for name, value in metadata.items():
            if '\n' in value or '\r' in value:
                raise ValueError('Newlines are not allowed in build environment values')
            destination.write(f'{name}={value}\n')

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--target')
    parser.add_argument('--helper-only', action='store_true')
    parser.add_argument('--github-env', action='store_true', help='Export public pack metadata and test paths to GITHUB_ENV')
    args = parser.parse_args()
    version = tomllib.loads((ROOT/'Cargo.toml').read_text(encoding='utf-8'))['package']['version']
    target = args.target
    os_name, arch = platform_id(target)
    system = f'{os_name}-{arch}'
    profile = 'release' if args.release else 'debug'
    output = ROOT/'target'/'optional-packs'/system/profile
    output.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    signing_key = env.pop("PAINTFE_PACK_SIGNING_KEY", "")
    if os.environ.get("CI") and args.release and not signing_key:
        raise SystemExit("Set the OPTIONAL_PACK_SIGNING_KEY repository secret before publishing optional packs.")
    # Keep one CMake job and reuse an existing compatible native build cache on this PC.
    env['CMAKE_BUILD_PARALLEL_LEVEL']='1'
    command = ['cargo','build','-j','1','--manifest-path',str(ROOT/'optional/jxl-encoder/Cargo.toml'),'--target-dir',str(ROOT/'target')]
    extra = (['--release'] if args.release else []) + (['--target',target] if target else [])
    subprocess.run(command+extra,cwd=ROOT,env=env,check=True)
    binary_root = ROOT/'target'
    if target: binary_root /= target
    binary_root /= profile
    suffix = '.exe' if os_name=='windows' else ''
    helper = binary_root/f'paintfe-jxl-encoder{suffix}'
    if signing_key:
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(signing_key))
    else:
        if os.environ.get('CI') and args.release:
            raise SystemExit('Set the OPTIONAL_PACK_SIGNING_KEY repository secret before publishing optional packs.')
        # Local review builds use an ephemeral signing key; official releases use
        # the same repository secret so already-installed packs survive app updates.
        key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex()
    (output/'public-key.hex').write_text(public,encoding='ascii')
    pack_paths = {}
    for pack, payload, license_path in [
        ('cjk', ROOT/'assets/fonts/NotoSansCJK-Regular.otf', ROOT/'assets/fonts/NotoSansCJK-OFL.txt'),
        ('jxl', helper, ROOT/'optional/jxl-encoder/LICENSE.txt'),
    ]:
        data = {payload.name:payload.read_bytes(),'LICENSE.txt':license_path.read_bytes()}
        manifest = json.dumps(dict(schema=1,pack=pack,app_version=version,platform=system,protocol=1,
            files=[dict(name=name,sha256=hashlib.sha256(contents).hexdigest(),size=len(contents)) for name,contents in data.items()]),separators=(',',':'))
        envelope = json.dumps(dict(manifest=manifest,signature=key.sign(manifest.encode()).hex()),separators=(',',':')).encode()
        path=output/f'PaintFE-pack-{pack}-{system}-v{version}.pfepack'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            archive.writestr('manifest.json',envelope)
            for name,contents in data.items(): archive.writestr(name,contents)
        pack_paths[pack] = path
    # Public trust anchor and pack sizes are compiled into this exact core build.
    # Signing keys are never written to disk, logs, environment or GitHub outputs.
    env['PAINTFE_PACK_PUBLIC_KEY']=public
    for pack,path in pack_paths.items(): env[f'PAINTFE_{pack.upper()}_PACK_SIZE']=str(path.stat().st_size)
    (output/'build-env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('PAINTFE_') and k.endswith(('_PUBLIC_KEY','_PACK_SIZE'))}),encoding='utf-8')
    if args.github_env:
        export_test_environment(output, pack_paths)
    if not args.helper_only:
        subprocess.run(['cargo','build','-j','1','--bin','PaintFE']+extra,cwd=ROOT,env=env,check=True)
        shutil.copy2(binary_root/f'PaintFE{suffix}',output/f'PaintFE{suffix}')
    for name,path in pack_paths.items(): print(f'{name}: {path.name}, {path.stat().st_size:,} bytes')
    print(f'Public build metadata: {output / "build-env.json"}')
if __name__ == '__main__': run()
