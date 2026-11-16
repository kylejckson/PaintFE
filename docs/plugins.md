# Script plugins

Script plugins add local raster effects to the Filter menu. They are portable Rhai scripts, not native DLLs. The current stable release may not include the browser yet.

## Finding and installing packs

Open **View > Community add-ons**, or the link in **Preferences > Optional features**. Opening the browser only reads local files. Click **Fetch add-ons** to download the community catalogue and preview artwork. Install or update a pack explicitly. Processing remains offline, and the app sends no documents or usage telemetry.

The catalogue is maintained in [PaintFE_plugins](https://github.com/kylejckson/PaintFE_plugins). Popular uses GitHub release download counts, including reinstalls and test downloads. Trending uses changes in those counts over the available window, up to seven days. Counts are not unique users or ratings. Refresh is manual; statistics are snapshots, not live totals.

Search, category filters, favourites and Installed help organise packs. Details include the source, discussion and report links. Those links open GitHub in your browser; PaintFE does not post comments or reports. Offline import accepts a `.pfeplugin` file through **Import pack**. Community signatures and every resource checksum are verified. Explicitly imported unsigned local packs are identified separately.

## Applying an effect

Open an image and choose an installed effect from its Filter submenu. Adjust its native controls, preview the result and use **Apply** or **Cancel**. Parameter changes always start from the original input, so previews do not accumulate. Cancelling discards the result. Apply creates one history entry and can be undone normally.

The host owns the dialog and cancellation. Scripts run in a separate process with resource and time limits. They cannot request downloads, open arbitrary files or resize the document through this plugin API. A failed or cancelled result does not replace document pixels. Stale results are rejected if the document or target layer changed.

API 1 supports 8-bit raster layers up to 16 million pixels. Deep-colour layers, text layers, mask editing and document transforms need other editing paths. Selection isolation is enforced by the host. Run ordinary scripts in the Script Editor for supported document-transform macros.

## Managing and authoring

In pack details, enable or disable individual effects, choose their menu category, set a local pack name, favourite the pack, export it, reinstall a damaged community pack or roll back to its previous installed version. Removing a pack removes its managed files and menu entries; separate local copies are preserved. A pack missing from a refreshed catalogue remains installed until you decide what to do with it.

**Create plugin** opens the existing Script Editor with plugin properties. Define the name, menu category and parameters, then save. A pack can contain multiple effects. Parameters support numbers, integers, switches, text, choices and colours, with optional section headings and a histogram. Scripts read these values through the `params` map.

**Edit local copy** preserves the signed original and creates an independent local draft. Save updates that draft; Save As creates another identity. Export produces a portable unsigned pack. Renaming or moving an installed effect changes local presentation, not its upstream source. See the [author guides](https://github.com/kylejckson/PaintFE_plugins/tree/main/docs) for the API, artwork requirements, packaging, licences and contribution process.

Native plugin libraries are not loaded. Local MCP agent control is a separate opt-in feature, described in [Local AI agent control](automation.md).

The browser uses Discover, Installed and Favourites views. Its empty Discover page explains the explicit GitHub connection; Fetch add-ons starts it immediately, without a second confirmation. Search sits beside sorting. Plugin properties groups metadata, effects and controls, and artwork into separate tabs. Edit code returns to the existing Script Editor and preserves the local draft. Properties blocks document input while open.

Cards wrap within the browser width and scroll vertically. Fetch shows the signed catalogue first, then retrieves initial artwork progressively with loading feedback and cancellation. Disk reads, checksums, decoding and resizing for browser previews run in bounded background batches; the UI uploads at most one prepared preview per frame. Browsing cached or installed items does not start network requests. Additional artwork can be requested from an item's details.

Installed cards provide enable and management controls, including removal with a protected confirmation. Properties uses the shared dialog header and footer; closing it preserves the draft. Its scrolling form reserves the wheel for scrolling. The View menu and Preferences link open the same browser, with Preferences handing over focus when selected.

## Themes and icon packs

Community add-ons has All, Effects, Themes and Icons filters. Appearance packages use the same signed catalogue and resource verification as effects, but contain data only. Installing does not apply a theme. Preview and apply lets you keep it or restore the complete previous appearance. Icon archives pass the existing bounded icon importer. Missing icons keep the built-in fallback. Appearance packs never expose script controls or change agent permissions.

The starter effects are Two Tone, Paper Grain, Colour Offset and Soft Vignette. Original generated artwork is paired with previews rendered by the actual scripts. The earlier Workshop packs remain authoring examples; removing their catalogue listings does not remove existing installed copies.

Community icon archives are limited to 1024 entries, 16 MiB expanded files and 64 MiB decoded pixels, with each PNG at most 1024 by 1024. Theme text is limited to 512 KiB. These are community-install limits and do not change the manual icon importer.
