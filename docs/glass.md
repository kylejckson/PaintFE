# Desktop glass

Preferences > Interface > Glass and transparency controls optional translucent UI fills. Glass is off by default. Enabling it allows the desktop to show through the workspace and translucent toolbars and panels. Document pixels, document transparency and exported files remain unchanged.

Workspace opacity controls the tint over the desktop outside the document. Panels/dialogs, headers, menus/popups, toolbars and buttons have separate opacity, tint and border controls. Text retains its normal alpha. Dialog headers use one uniform accent fill over their containing dialog, without an image-textured cutout. Nested opaque groups keep their original fill.

Surface tint changes RGB only. Existing background alpha and the surface opacity multiply, including the tool options bar's Background opacity. A custom tint cannot replace a lower alpha with an opaque fill. Border strength is controlled separately.

The fast path paints ordinary tinted rectangles using the existing renderer. It does not composite document pixels, create backdrop textures or run CPU blur filters. Native desktop blur is a separate optional switch and defaults off. Its strength and performance depend on the operating system. Leave it off for the lightest desktop transparency.

Windows uses desktop transparency and optional native Desktop Acrylic where available. Windows 11 build 22621 and newer use the documented DWM system backdrop; older Windows versions use the legacy Acrylic policy. macOS uses optional native visual effects. Linux desktop transparency and blur depend on the compositor, with an ordinary opaque appearance available by disabling glass. Native effects on macOS/Linux have not been visually validated on this Windows development machine.

Reset to Signal restores the default light Signal appearance, disables glass and removes previous colour and geometry overrides. Switching Light/Dark through Preferences or View adopts the selected mode's default colours and clears fixed glass tints, while retaining glass enablement, native blur, opacity and layout choices. Imported theme palettes are retained until an explicit mode switch or reset.

Surface settings are exported with themes. Automation permissions are not. Community themes install first, then offer Preview and apply with Keep appearance or Restore previous appearance. Glass Dark uses charcoal surfaces and ice-blue accents. Glass Light uses pearl-white surfaces and cool blue accents. Both request native desktop blur, keep dialog headers uniform, and give controls a more opaque fill for readability. Native blur can be disabled independently for lighter desktop transparency. No theme is fetched or applied automatically.

The small documented egui frame hook changes only outer-container frame shapes. It leaves layout, focus, pointer handling and document rendering intact. Removed experimental canvas-sampling blur fields in older local themes are ignored safely when importing them.

## Windows backing surface

GPU alpha support alone is insufficient. The Windows adapter installs a stock zero-alpha black class brush, clears the client redirection surface once when transparency changes, and explicitly enables DWM alpha composition. Plain transparency does not clear the accent policy unnecessarily. These calls run on a state change, not per frame. Disabling glass restores opaque composition; re-enabling it clears and restores the native alpha path. The existing Vulkan backend is retained.

The opt-in `transparency_surface` example reports actual surface alpha capabilities for Vulkan, ordinary DX12 and DirectComposition on this machine. An advertised premultiplied mode is not proof that desktop pixels are visible. The original gray-background reproduction was verified in the actual app against a separate native checkerboard window, including glass off/on and a standard New File dialog. The user independently confirmed desktop visibility. See [Win32 reference demos](https://github.com/jeweg/win32-window-transparency) for the redirection-surface explanation.
