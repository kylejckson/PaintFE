import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("codec_boundary", Path(__file__).resolve().parents[1] / "check_codec_boundary.py")
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)


class CodecBoundaryTests(unittest.TestCase):
    def test_container_parser_allowed_but_aliased_codec_and_locked_codec_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "Cargo.toml"
            manifest.write_text('[dependencies]\nheifer-isobmff="=0.2.0"\n')
            self.assertEqual(boundary.check(root), [])
            manifest.write_text('[target.windows.dependencies]\ncodec={package="heifer-hevc-dec", version="0.2"}\n')
            self.assertEqual(len(boundary.check(root)), 1)
            manifest.write_text('[dependencies]\nheifer-isobmff="=0.2.0"\n')
            (root / "Cargo.lock").write_text('[[package]]\nname="heifer-hevc-enc"\nversion="0.2.0"\n')
            self.assertEqual(len(boundary.check(root)), 1)

    def test_repository_boundary(self):
        self.assertEqual(boundary.check(), [])
