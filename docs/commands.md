# Built-in commands and shortcuts

`src/config/commands.rs` contains stable command identities and translated label,
category and document requirements. `src/app/commands.rs` owns availability and
execution. Menus and keyboard dispatch share these handlers. Existing saved action
names remain valid; additional actions use `Menu(CommandName)` identities.

Preferences > Keybindings includes built-in menu commands, guide commands and the
remaining tools. Crop to Selection defaults to Ctrl+Shift+X, clockwise canvas
rotation to Ctrl+H and anticlockwise rotation to Ctrl+G. Explicitly saved bindings
take priority over these newly introduced defaults. Commands can be rebound or
unbound. Preferences groups rows by stable category keys, independently of saved
configuration order, and each row uses its action identity. Toolbar hints use current bindings rather than fixed shortcut strings.

Document commands remain unavailable without a document. Selection commands retain
their selection requirements. Modal dialogs, key capture and focused text/numeric
fields own their keys. New command shortcuts do not interrupt canvas text editing.
Clipboard platform-event handling remains separate because operating systems can
supply clipboard events as well as ordinary key presses.

The catalogue covers fixed application commands. Recent filenames, installed user
scripts, plugin entries and individual parameter choices are dynamic menu data,
not separate built-in commands. They retain their existing navigation.

When adding a fixed command, register its metadata and binding identity, implement
availability and execution centrally, and route its menu through that handler.
Keep identity independent of translated labels. Add a regression for eligibility,
settings round trips and the actual effect or dialog it opens.

The base `en` catalogue uses Australian spelling. Its stable language ID and
translation keys remain unchanged. Legacy English display labels used by enums
continue to resolve through aliases, including personal label overrides. Language
picker entries draw the native name and parenthesised language code separately to
avoid joining Arabic and Latin text into one ambiguous bidirectional run.

Hold Ctrl with Brush, Pencil, Eraser, Line, Fill or Smudge to sample a colour temporarily.
Left-click samples the primary colour and right-click samples the secondary colour.
Release the modifier to return to the previous tool. Preferences > Keybinds > Tools
includes Temporary colour picker modifier, with Ctrl, Alt and Disabled choices.
The configured Brush/Eraser resize modifier takes priority when both modifiers match.
Other shortcuts, text fields, dialogs and Clone Stamp source placement keep their
existing input. A held paint stroke finishes before temporary sampling can start.

The Eraser shortcut (E by default) toggles between Eraser and the previous tool.
Toolbar tool selection still selects a tool directly. Rebinding or disabling the
Eraser shortcut applies to the toggle too. A toggle during painting waits for the
stroke to finish, and repeated key events do not toggle again.

## Selection rotation

With a selection, Ctrl+G and Ctrl+H lift the active layer's selected pixels into
one editable transform with handles. Repeating a rotation changes that same
floating selection, without cutting or resampling the layer again. Enter or Apply
commits one history entry. Escape or Cancel restores the original pixels and
selection. Whole-canvas rotation remains available when there is no selection.
Other image edits wait until the floating selection is applied or cancelled.
Pending-transform Undo follows Preferences > Editing > Undo tool adjustments.

The Line tool supports temporary Ctrl colour sampling too. Its two blue curve
handles start apart, and changing colour, size, pattern or blending updates the
pending line while the pointer remains over the controls. Apply and Cancel in
the options bar perform the same actions as Enter and Escape.
