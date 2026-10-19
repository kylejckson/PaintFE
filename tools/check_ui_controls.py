#!/usr/bin/env python3
"""Check the shared numeric-control boundary, without executing application code.

Run: python tools/check_ui_controls.py --self-test --report target/ui-controls.json
This token audit catches aliases/imports as well as qualified constructors. Custom
2D canvas editors still need review; this is not a proof about arbitrary paint code.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def tokens(source):
    """Rust identifiers and punctuation, skipping strings and nested comments."""
    result = []
    i = 0
    line = 1
    raw = re.compile(r'(?:br|cr|r)(#*)"')
    char = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'")
    word = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|::|[^\s]")
    while i < len(source):
        previous = i
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(source) and depth:
                if source.startswith("/*", i):
                    depth += 1
                    i += 2
                elif source.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
        elif match := raw.match(source, i):
            terminator = '"' + match[1]
            end = source.find(terminator, match.end())
            i = len(source) if end < 0 else end + len(terminator)
        elif source[i] == '"':
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif match := char.match(source, i):
            i = match.end()
        elif match := word.match(source, i):
            result.append((match[0], line))
            i = match.end()
        else:
            i += 1
        line += source.count("\n", previous, i)
    return result


def violations(source, filename):
    items = tokens(source)
    errors = []
    for i, (token, line) in enumerate(items):
        if filename == "src/ui/numeric.rs" and token == "DragValue":
            continue
        if filename == "src/ui/controls/slider.rs" and token == "Slider":
            if i >= 2 and items[i - 2][0] == "egui":
                tail = [text for text, _ in items[i + 1:]]
                depth = 0
                end = None
                for j, part in enumerate(tail):
                    if part == "(":
                        depth += 1
                    elif part == ")":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                if end is None or tail[end + 1:end + 7] != [".", "show_value", "(", "false", ")", "."]:
                    errors.append((line, "native slider must immediately disable its built-in editor"))
            continue
        if token in {"DragValue", "DragValueSettings"}:
            errors.append((line, "raw numeric editor outside the shared implementation"))
        if token == "Slider":
            prefix = [text for text, _ in items[max(0, i - 4):i]]
            if prefix != ["ui", "::", "controls", "::"] and filename != "src/ui/controls/mod.rs" and not (filename.startswith("src/ui/controls/") and (i == 0 or items[i - 1][0] != "::")):
                errors.append((line, "use crate::ui::controls::Slider, including in tests"))
        if token == "Numeric" and i >= 2 and items[i - 2][0] == "numeric" and filename != "src/ui/controls/mod.rs":
            errors.append((line, "import Numeric through ui::controls"))
        if (token.startswith("color_edit_button") or token in {"color_picker_hsva_2d", "color_picker_color32"}) and filename != "src/ui/controls/color.rs":
            prefix = [text for text, _ in items[max(0, i - 4):i]]
            if prefix != ["ui", "::", "controls", "::"] and filename != "src/ui/controls/mod.rs":
                errors.append((line, "native colour picker bypasses shared channel editors"))
        if token == "TextEdit":
            # Scalar dimensions belong to Numeric; genuine string editors such as
            # hex colours and W:H aspect-ratio expressions are intentionally separate.
            nearby = [text for text, _ in items[i:i + 14]]
            if "width_input" in nearby or "height_input" in nearby:
                errors.append((line, "dimension text editor bypasses Numeric::dimension"))
    return errors


class BoundaryTests(unittest.TestCase):
    def test_rejects_constructor_import_alias_and_type_alias(self):
        for source in ["egui::Slider::new(v, r)", "use egui::{Slider as Other};",
                       "type Other = e::widgets::DragValue;", "e::Slider::from_get_set(r, f)",
                       "use egui::DragValueSettings;", "ui::numeric::Numeric::new(v)",
                       "ui.color_edit_button_srgba(v)", "egui::color_picker::color_picker_hsva_2d(ui, v, a)"]:
            self.assertTrue(violations(source, "src/example.rs"), source)

    def test_ignores_strings_characters_and_nested_comments(self):
        self.assertFalse(violations('/* Slider /* DragValue */ */ // Slider\n'
                                    'let s = r###"Slider"###; let c = \'"\';', "src/example.rs"))

    def test_accepts_canonical_path_and_rejects_dimension_text(self):
        self.assertFalse(violations("crate::ui::controls::Slider::new(v, r)", "src/example.rs"))
        self.assertTrue(violations("egui::TextEdit::singleline(&mut self.width_input)", "src/example.rs"))

    def test_native_slider_cannot_reenable_its_private_editor(self):
        name = "src/ui/controls/slider.rs"
        self.assertFalse(violations("egui::Slider::new(v, r).show_value(false).step_by(1.0)", name))
        self.assertTrue(violations("egui::Slider::new(v, r).show_value(true).step_by(1.0)", name))
        self.assertTrue(violations("egui::DragValue::new(v)", name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests))
        if not result.wasSuccessful():
            return 1
    errors = []
    inventory = []
    for folder in ("src", "tests", "examples"):
        for path in sorted((ROOT / folder).rglob("*.rs")):
            name = path.relative_to(ROOT).as_posix()
            source = path.read_text(encoding="utf-8")
            errors += [f"{name}:{line}: {message}" for line, message in violations(source, name)]
            counts = collections.Counter(text for text, _ in tokens(source)
                                         if text in {"Numeric", "Slider", "GradientSlider"})
            if counts:
                inventory.append({"file": name, "references": dict(counts)})
    report = {"errors": errors, "files": inventory}
    if args.report:
        destination = ROOT / args.report
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for error in errors:
        print(error)
    print(f"Shared control audit: {len(errors)} violations; {len(inventory)} files inventoried.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
