# Arabic translation

The contributor's `ar.txt` catalogue is embedded as `locales/ar.txt`, with five
newer JPEG XL and slider-reset labels added for current catalogue parity. Its
1556 keys retain the same interpolation placeholders as English. Arabic (`ar`)
is available in the language picker and through system-locale detection.

Noto Sans Arabic is bundled for offline proportional and monospace UI fallback;
its original SIL Open Font License is shipped beside it. The locale audit checks
every translated letter and combining mark against the actual bundled fonts.

This adds the translation and font, not a mirrored application layout. Egui 0.35
shapes glyph runs but does not implement full bidirectional paragraph layout.
Mixed Arabic/Latin text, numbers, punctuation, wrapping, editing and selection
need native-speaker testing. Do not advertise full RTL support based on a passing
catalogue audit. Translation content remains in logical Unicode order; no manual
reversal or presentation-form substitutions are applied.

Review the menus, Preferences, dialogs and tooltips in both themes and at larger
text sizes. In particular, check messages containing filenames, keyboard shortcuts,
numeric values and English technical terms. Report screenshots with the affected
translation key where possible.
