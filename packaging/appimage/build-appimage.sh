#!/usr/bin/env bash
# build-appimage.sh — Build a PaintFE AppImage
# Run from the repo root: bash packaging/appimage/build-appimage.sh
#
# Prerequisites (installed by this script if missing):
#   cargo, linuxdeploy
# -----------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APPDIR="$REPO_ROOT/packaging/appimage/PaintFE.AppDir"
TOOLS_DIR="$REPO_ROOT/packaging/.tools"
mkdir -p "$TOOLS_DIR"

BIN="$REPO_ROOT/target/release/PaintFE"
if [ -f "$BIN" ]; then
  echo "==> [1/4] Release binary already built — skipping cargo build."
else
  echo "==> [1/4] Building release binary..."
  cd "$REPO_ROOT"
  cargo build --release
fi

if [ ! -f "$BIN" ]; then
  echo "ERROR: release binary not found at $BIN"; exit 1
fi

echo "==> [2/4] Staging AppDir..."
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp "$BIN" "$APPDIR/usr/bin/PaintFE"
chmod +x "$APPDIR/usr/bin/PaintFE"
chmod +x "$APPDIR/AppRun"

# Copy icon — adjust path if your icon asset differs
ICON="$REPO_ROOT/assets/icons/app_icon.png"
if [ -f "$ICON" ]; then
  cp "$ICON" "$APPDIR/PaintFE.png"
  cp "$ICON" "$APPDIR/usr/share/icons/hicolor/256x256/apps/PaintFE.png"
else
  echo "WARN: icon not found at $ICON — AppImage will lack an icon"
fi

echo "==> [3/4] Fetching linuxdeploy (if needed)..."
LINUXDEPLOY="$TOOLS_DIR/linuxdeploy-x86_64.AppImage"
if [ ! -f "$LINUXDEPLOY" ]; then
  curl -L -o "$LINUXDEPLOY" \
    "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
  chmod +x "$LINUXDEPLOY"
fi

echo "==> [4/4] Packaging AppImage..."
cd "$REPO_ROOT"
ARCH=x86_64 "$LINUXDEPLOY" \
  --appdir "$APPDIR" \
  --desktop-file "$APPDIR/PaintFE.desktop" \
  --output appimage

echo ""
echo "Done! AppImage written to: $(ls "$REPO_ROOT"/PaintFE*.AppImage 2>/dev/null | tail -1)"
echo ""
echo "NOTE: wgpu requires either Vulkan (via host driver ICDs) or Mesa/OpenGL."
echo "      libvulkan.so.1 is intentionally NOT bundled — it must come from the host."
