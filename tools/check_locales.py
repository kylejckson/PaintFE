"""Validate maintained PaintFE catalogs without building or launching the app.

Run from any directory: python tools/check_locales.py
Only Python's standard library is required. No network access.
"""
from collections import Counter
from pathlib import Path
import re
import struct
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
MAINTAINED = tuple(sorted(path.stem for path in (ROOT / "locales").glob("*.txt") if path.stem != "en"))
errors = []


def read_catalog(code):
    result = {}
    for number, line in enumerate((ROOT / "locales" / f"{code}.txt").read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"{code}:{number}: missing '='")
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in result:
            errors.append(f"{code}:{number}: duplicate {key}")
        if not value:
            errors.append(f"{code}:{number}: empty {key}")
        if "\ufffd" in value:
            errors.append(f"{code}:{number}: invalid Unicode in {key}")
        if unicodedata.normalize("NFC", value) != value:
            errors.append(f"{code}:{number}: non-normalized Unicode in {key}")
        forbidden = {"uk": r"[ыэёъЫЭЁЪ]", "be": r"[иИщЩъЪїЇєЄ]", "ro": r"[şţŞŢ]"}.get(code)
        if forbidden and re.search(forbidden, value):
            errors.append(f"{code}:{number}: unexpected alphabet character in {key}")
        result[key] = value
    return result


def placeholders(value):
    return Counter(re.findall(r"\{[^{}]*\}", value))


def font_has_glyph(path):
    data = path.read_bytes()
    u16 = lambda offset: struct.unpack_from(">H", data, offset)[0]
    u32 = lambda offset: struct.unpack_from(">I", data, offset)[0]
    cmap = next(u32(12 + i * 16 + 8) for i in range(u16(4)) if data[12 + i * 16:16 + i * 16] == b"cmap")
    tables = [cmap + u32(cmap + 4 + i * 8 + 4) for i in range(u16(cmap + 2))]

    def has(code):
        for start in tables:
            kind = u16(start)
            if kind == 12:
                for index in range(u32(start + 12)):
                    first, last, glyph = struct.unpack_from(">III", data, start + 16 + index * 12)
                    if first <= code <= last and glyph + code - first:
                        return True
            elif kind == 4 and code <= 0xFFFF:
                count = u16(start + 6) // 2
                for index in range(count):
                    last = u16(start + 14 + index * 2)
                    first = u16(start + 16 + count * 2 + index * 2)
                    if first <= code <= last:
                        delta = u16(start + 16 + count * 4 + index * 2)
                        position = start + 16 + count * 6 + index * 2
                        offset = u16(position)
                        glyph = u16(position + offset + 2 * (code - first)) if offset else code
                        if glyph and (glyph + delta) & 0xFFFF:
                            return True
        return False

    return has


english = read_catalog("en")
catalogs = {code: read_catalog(code) for code in MAINTAINED}
for code, catalog in catalogs.items():
    missing = english.keys() - catalog.keys()
    extra = catalog.keys() - english.keys()
    errors.extend(f"{code}: missing {key}" for key in sorted(missing))
    errors.extend(f"{code}: extra {key}" for key in sorted(extra))
    for key in english.keys() & catalog.keys():
        if placeholders(english[key]) != placeholders(catalog[key]):
            errors.append(f"{code}: placeholder mismatch in {key}")
        functions = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\(\)", english[key])
        if any(catalog[key].count(name) != english[key].count(name) for name in functions):
            errors.append(f"{code}: translated code identifier in {key}")
    print(f"{code}: {len(catalog)}/{len(english)} keys, {len(missing)} missing, {len(extra)} extra")

used = set()
for path in (ROOT / "src").rglob("*.rs"):
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"(?m)^\s*//.*$", "", source)
    used.update(re.findall(r'\bt!\(\s*"([a-z][a-z0-9_.]+)"', source))
    used.update(re.findall(r'format_translation\(\s*"([a-z][a-z0-9_.]+)"', source))
errors.extend(f"source lookup missing from English: {key}" for key in sorted(used - english.keys()))

# Core fallback families plus the optional pack source. Check both, without assuming system fonts.
fonts = ROOT / "assets/fonts"
proportional = [font_has_glyph(fonts / name) for name in ("DMSans-Regular.ttf", "NotoSans-Regular.ttf")]
monospace = [font_has_glyph(fonts / name) for name in ("JetBrainsMono-Regular.ttf", "NotoSans-Regular.ttf")]
for name in ("NotoSansCJK-Regular.otf", "NotoSansThai.ttf", "NotoSansArabic.ttf", "CJK-Picker.otf"):
    check = font_has_glyph(fonts / name)
    proportional.append(check)
    monospace.append(check)
# Combining marks are essential for Vietnamese and Thai, even when not alphabetic.
letters = {character for catalog in catalogs.values() for value in catalog.values() for character in value if character.isalpha() or unicodedata.category(character).startswith("M")}
registered = set(re.findall(r'^\s*\("([^"]+)",\s*"', (ROOT / "src/i18n.rs").read_text(encoding="utf-8").split('/// Initialize')[0], re.M))
errors.extend(f"catalog not registered: {code}" for code in set(MAINTAINED) | {"en"} if code not in registered)
errors.extend(f"registered language missing catalog: {code}" for code in registered if not (ROOT / "locales" / f"{code}.txt").exists())
embedded = set(re.findall(r'include_str!\("\.\./locales/([^"/]+)\.txt"\)', (ROOT / "src/i18n.rs").read_text(encoding="utf-8").split('#[cfg(test)]')[0]))
errors.extend(f"registered language not embedded: {code}" for code in registered - embedded)
for family, checks in (("proportional", proportional), ("monospace", monospace)):
    absent = sorted(character for character in letters if not any(check(ord(character)) for check in checks))
    if absent:
        errors.append(f"{family}: missing letter glyphs: {''.join(absent)}")
    print(f"{family}: checked {len(letters)} distinct letters")

# A tiny subset keeps all language-picker names readable before installing CJK.
picker = font_has_glyph(fonts / "CJK-Picker.otf")
for character in "日本語中文简体繁體한국어":
    if not picker(ord(character)):
        errors.append(f"language picker missing {character}")
bootstrap = (ROOT / "src/app/bootstrap.rs").read_text(encoding="utf-8")
native_bootstrap = re.sub(r'(?s)#\[cfg\(target_arch = "wasm32"\)\]\s*\(.*?\),', "", bootstrap)
if 'include_bytes!("../../assets/fonts/NotoSansCJK-Regular.otf")' in native_bootstrap:
    errors.append("full optional CJK font must not be embedded in the core")
if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"PASS: {len(MAINTAINED)} complete catalogs; {len(used)} literal source lookups; placeholders and letter glyphs valid.")
