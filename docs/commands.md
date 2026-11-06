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
