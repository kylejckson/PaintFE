# PaintFE patch to rfd 0.12.1

Upstream: https://github.com/PolyMeilex/rfd, crate release 0.12.1.
Original source and licences are retained.

Add `FileDialog::set_add_to_recent` to suppress automatic Windows recent-file
registration for configuration resources. Default behaviour remains unchanged.
Preserve existing native options when adding flags, including multi-selection,
folder selection, overwrite prompts and FOS_DONTADDTORECENT (0x02000000).
Other backends ignore the new option.

On upgrades, check whether upstream exposes this policy, rebase only if needed,
and test single/multiple opens, folders and saves on Windows. Verify document
history still works and model/runtime selection does not enter Jump Lists.
