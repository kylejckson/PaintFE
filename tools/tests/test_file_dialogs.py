import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("file_dialogs", Path(__file__).resolve().parents[1] / "check_file_dialogs.py")
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)


class FileDialogBoundaryTests(unittest.TestCase):
    def test_raw_pickers_rejected_outside_shared_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/services").mkdir(parents=True)
            (root / "src/services/file_dialogs.rs").write_text("rfd::FileDialog::new();")
            self.assertEqual(boundary.check(root), [])
            (root / "src/picker.rs").write_text("rfd::FileDialog::new();\nrfd::AsyncFileDialog::default();")
            self.assertEqual(len(boundary.check(root)), 2)

    def test_repository_boundary(self):
        self.assertEqual(boundary.check(), [])
