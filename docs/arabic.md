# Arabic Translation

The contributor's `ar.txt` catalog is embedded as `locales/ar.txt`, with newer labels added to keep it aligned with the current catalog. Its keys keep the same interpolation placeholders as English. Arabic (`ar`) is available in the language picker and through system-locale detection.

Noto Sans Arabic is bundled for offline proportional and monospace UI fallback. Its original SIL Open Font License is included alongside it. The locale audit checks each translated letter and combining mark against the actual bundled fonts.

The shared egui/epaint text pipeline now includes a focused bidi, shaping-cluster, and logical-cursor fix; see [RTL UI text](rtl-text.md). Translation strings remain in logical Unicode order. This changes text direction, not the whole application layout. Mixed text and language quality still need native-speaker review.

Review the menus, Preferences, dialogs, and tooltips in both themes and at larger text sizes. In particular, check messages that contain filenames, keyboard shortcuts, numeric values, and English technical terms. If possible, include screenshots showing the affected translation key.
