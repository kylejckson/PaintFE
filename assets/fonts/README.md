# UI font fallbacks

PaintFE embeds its UI fonts and never downloads fonts at runtime.

`NotoSansCJK-Regular.otf` and `NotoSansThai.ttf` come from the Noto and Google Fonts repositories:

- https://github.com/notofonts/noto-cjk/tree/main/Sans
- https://github.com/google/fonts/tree/main/ofl/notosansthai

Their original SIL Open Font License files are included beside the fonts.
Chinese, Japanese and Korean glyphs and Thai combining marks are available to proportional text,
monospace section labels and panel titles. An available system CJK font stays
ahead of the bundled CJK fallback to preserve regional Japanese/Chinese
glyph forms. Both bundled fallbacks are available immediately when changing
languages.

Run `python tools/check_locales.py` from the repository root to check catalog
letters and combining marks against the bundled fonts.
