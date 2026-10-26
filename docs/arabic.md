# Arabic translation

The contributor's `ar.txt` catalogue is embedded as `locales/ar.txt`, with
newer labels added for current catalogue parity. Its keys retain the same
interpolation placeholders as English. Arabic (`ar`)
is available in the language picker and through system-locale detection.

Noto Sans Arabic is bundled for offline proportional and monospace UI fallback;
its original SIL Open Font License is shipped beside it. The locale audit checks
every translated letter and combining mark against the actual bundled fonts.

The shared egui/epaint text pipeline now includes a focused bidi, shaping-cluster
and logical-cursor patch; see [RTL UI text](rtl-text.md). Translation strings stay
in logical Unicode order. This changes text direction, not the whole application
layout. Mixed text and language quality still need native-speaker review.

Review the menus, Preferences, dialogs and tooltips in both themes and at larger
text sizes. In particular, check messages containing filenames, keyboard shortcuts,
numeric values and English technical terms. Report screenshots with the affected
translation key where possible.
