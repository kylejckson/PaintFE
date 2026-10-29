"""Native file pickers must declare their document/resource purpose centrally."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"rfd\s*::\s*(?:Async)?FileDialog\s*::\s*(?:new|default)\s*\(")


def check(root=ROOT):
    errors = []
    for path in (root / "src").rglob("*.rs"):
        if path.relative_to(root).as_posix() == "src/services/file_dialogs.rs":
            continue
        text = path.read_text(encoding="utf-8")
        for match in PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"{path.relative_to(root)}:{line}: use services::file_dialogs with an explicit purpose")
    return errors


if __name__ == "__main__":
    errors = check()
    print("\n".join(errors) if errors else "Native picker boundary passed.")
    raise SystemExit(bool(errors))
