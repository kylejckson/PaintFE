# Personal interface labels

Open **Preferences > General > Language > Customize labels**. Search using the
current wording or a stable key such as `menu.view`, select a row, and edit its
replacement. **Apply customizations** validates and saves all staged changes.
Closing without applying offers to discard edits.

The original bundled catalogue is preserved. Reset a label or stage a full reset,
then Apply. Each language has independent customizations. Identical words in
different contexts can have different keys: the View menu and the View shortcut
category are separate entries. User document names, shortcut identities and saved
setting values are not renamed.

The list searches original text, personal replacements and keys. All/Customized
filters and a result count help narrow the list. Wording and technical keys have
separate visual hierarchy, without em-dash separators. A list/detail layout
stacks vertically on smaller windows, and only visible result rows are rendered.
The editor can be moved using its header and resized from its edges. Its size is
capped to the app viewport, with the bottom Apply/Close actions outside the
scrolling body. Preferences search also sees customized text. Empty values,
unsupported control characters, excessively long labels and changed/missing
named placeholders are rejected before saving.

**Import / export** shares only replacements as JSON. Export copies JSON to the
clipboard; paste JSON into the box and Import to merge it into the draft. Import
does not apply changes until Apply is pressed. Unknown keys and incompatible
placeholders are rejected. The maximum customization file size is 1 MiB.

Desktop files live in `labels/<language>.json` under PaintFE's active data folder
(inside `paintfe_data` in portable mode). Writes replace the file atomically, and
storage-mode migration includes this folder. Browser builds use localStorage.
On upgrades, removed keys or incompatible old replacements fall back to the
bundled text; the saved file is not silently rewritten. Other valid replacements
continue to load.

Most interface text uses the catalogue; diagnostic messages and a few legacy
hardcoded captions are not editable here. New editor/AVIF captions currently use
English defaults in the maintained catalogues, ready for translation.
