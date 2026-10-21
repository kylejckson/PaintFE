# UI font fallbacks

PaintFE embeds its UI fonts and never downloads fonts at runtime.

`NotoSansCJK-Regular.otf` and `NotoSansThai.ttf` come from the Noto and Google Fonts repositories:

- https://github.com/notofonts/noto-cjk/tree/main/Sans
- https://github.com/google/fonts/tree/main/ofl/notosansthai

Their original SIL Open Font License files are included beside the fonts.

`NotoSansArabic.ttf` is the unmodified variable Noto Sans Arabic font from
https://github.com/google/fonts/tree/main/ofl/notosansarabic, with its original
`NotoSansArabic-OFL.txt`. It is available in both UI font families; see
`docs/arabic.md` for the current bidirectional-layout limitations.
The bundled Arabic font's SHA-256 is
`63111b5b2e074dd48cc67692e0a2726d86ee94c1c37fe8598257b7b4e87e869e`.
Chinese, Japanese and Korean glyphs and Thai combining marks are available to proportional text,
monospace section labels and panel titles. An available system CJK font stays
ahead of the optional CJK fallback to preserve regional Japanese/Chinese
glyph forms. Both bundled fallbacks are available immediately when changing
languages.

Run `python tools/check_locales.py` from the repository root to check catalog
letters and combining marks against the bundled fonts.

## Optional CJK coverage

The full `NotoSansCJK-Regular.otf` is pack source, no longer embedded in the core.
`CJK-Picker.otf` is a renamed 9 KB subset used only for language-picker names.
Both are covered by `NotoSansCJK-OFL.txt`. System fonts or the signed optional
pack provide full Chinese, Japanese and Korean UI coverage.
