"""Release staging regressions using real signed archives and nested artifact paths."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

SCRIPT = Path(__file__).resolve().parents[1] / 'stage_release_packs.py'
spec = importlib.util.spec_from_file_location('staging', SCRIPT)
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


class ReleasePackTests(unittest.TestCase):
    def setUp(self):
        parent = SCRIPT.parents[1] / 'target' / 'release-pack-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.dest = self.root/'input', self.root/'packs'
        self.key = Ed25519PrivateKey.generate()
        public = self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        self.core = self.root/'core-fixture.bin'
        self.core.write_bytes(b'fixture-core\x00'+public.encode()+b'\x00')
        self.expected = staging.expected_packs('v1.4.13')
        self.paths = {}
        for name, (pack, platform) in self.expected.items():
            path = self.source/platform/'release'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            self.write_pack(path, pack, platform)
            self.paths[name] = path

    def write_pack(self, path, pack, platform, version='1.4.13', corrupt=False):
        payload = ('NotoSansCJK-Regular.otf' if pack == 'cjk' else
                   f'paintfe-{"avif-codec" if pack == "avif" else "heic-encoder" if pack == "heic" else "jxl-encoder"}' + ('.exe' if platform.startswith('windows-') else ''))
        data = {payload:b'fixture payload', 'LICENSE.txt':b'fixture license'}
        if pack == 'heic': data.update({'SOURCES.tar.gz':b'source archive','BUILDING.txt':b'build instructions'})
        manifest = json.dumps(dict(schema=1, protocol=1, app_version=version, pack=pack,
            platform=platform, files=[dict(name=n,size=len(b),sha256=hashlib.sha256(b).hexdigest())
                                      for n,b in data.items()]))
        envelope = dict(manifest=manifest, signature=self.key.sign(manifest.encode()).hex())
        if corrupt:
            data[payload] = b'altered payload'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('manifest.json',json.dumps(envelope))
            for name,value in data.items():
                archive.writestr(name,value)

    def stage(self):
        return staging.stage(self.source,self.dest,'v1.4.13',self.core)

    def test_nested_four_platform_artifacts_produce_exact_flat_sixteen_packs(self):
        self.assertEqual(self.stage(),sorted(self.expected))
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()),sorted(self.expected))
        for name,path in self.paths.items():
            self.assertEqual(path.read_bytes(),(self.dest/name).read_bytes())

    def test_missing_pack_fails_before_staging(self):
        next(iter(self.paths.values())).unlink()
        with self.assertRaisesRegex(ValueError,'sixteen unique'):
            self.stage()
        self.assertFalse(self.dest.exists())

    def test_duplicate_pack_fails_before_staging(self):
        path=next(iter(self.paths.values()))
        (self.source/path.name).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError,'sixteen unique'):
            self.stage()

    def test_unexpected_version_filename_fails(self):
        path=next(iter(self.paths.values()))
        path.rename(path.with_name(path.name.replace('v1.4.13','v1.4.12')))
        with self.assertRaises(ValueError):
            self.stage()

    def test_signed_wrong_version_or_platform_fails(self):
        name=next(iter(self.paths))
        pack,platform=self.expected[name]
        for version,identity in [('1.4.12',platform),('1.4.13','wrong-platform')]:
            with self.subTest(version=version,platform=identity):
                self.write_pack(self.paths[name],pack,identity,version=version)
                with self.assertRaisesRegex(ValueError,'Wrong pack identity'):
                    self.stage()

    def test_changed_payload_fails_before_staging(self):
        name=next(iter(self.paths))
        self.write_pack(self.paths[name],*self.expected[name],corrupt=True)
        with self.assertRaisesRegex(ValueError,'integrity failed'):
            self.stage()
        self.assertFalse(self.dest.exists())

    def test_pack_must_match_embedded_exe_key(self):
        self.core.write_bytes(b'not the matching core')
        with self.assertRaisesRegex(ValueError,'signing key'):
            self.stage()

    def test_existing_staging_is_not_overwritten(self):
        self.stage()
        with self.assertRaisesRegex(ValueError,'must be empty'):
            self.stage()

    def test_public_urls_and_bytes_are_verified_for_all_sixteen(self):
        self.stage()
        urls=[]
        def fetch(url,timeout):
            urls.append(url)
            return io.BytesIO((self.dest/url.rsplit('/',1)[1]).read_bytes())
        with patch.object(staging,'urlopen',side_effect=fetch):
            staging.verify_published(self.dest,'v1.4.13')
        self.assertEqual(len(urls),16)
        self.assertTrue(all('/releases/download/v1.4.13/' in u for u in urls))

    def test_public_404_and_wrong_bytes_fail(self):
        self.stage()
        with patch.object(staging,'urlopen',side_effect=HTTPError('url',404,'Missing',{},None)):
            with self.assertRaises(HTTPError):
                staging.verify_published(self.dest,'v1.4.13')
        with patch.object(staging,'urlopen',return_value=io.BytesIO(b'wrong file')):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                staging.verify_published(self.dest,'v1.4.13')


if __name__ == '__main__':
    unittest.main()
