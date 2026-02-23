#!/usr/bin/env bash
# build-flatpak.sh — Build and install a local PaintFE Flatpak
# Run from the repo root: bash packaging/flatpak/build-flatpak.sh
#
# Prerequisites (installed by this script if missing):
#   flatpak, flatpak-builder, python3
# -----------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLATPAK_DIR="$REPO_ROOT/packaging/flatpak"
BUILD_DIR="$REPO_ROOT/packaging/.flatpak-build"
GENERATOR="$FLATPAK_DIR/flatpak-cargo-generator.py"

# ── Step 1: install flatpak-builder if needed ───────────────
if ! command -v flatpak-builder &>/dev/null; then
  echo "==> Installing flatpak-builder..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get install -y flatpak flatpak-builder
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y flatpak flatpak-builder
  elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm flatpak flatpak-builder
  else
    echo "ERROR: Cannot auto-install flatpak-builder. Install it manually then re-run."; exit 1
  fi
fi

# ── Step 2: add Flathub remote and install Freedesktop SDK ──
flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install --user -y flathub \
  org.freedesktop.Platform//23.08 \
  org.freedesktop.Sdk//23.08 \
  org.freedesktop.Sdk.Extension.rust-stable//23.08

# ── Step 3: fetch flatpak-cargo-generator if needed ─────────
if [ ! -f "$GENERATOR" ]; then
  echo "==> Downloading flatpak-cargo-generator.py..."
  curl -L -o "$GENERATOR" \
    "https://raw.githubusercontent.com/flatpak/flatpak-builder-tools/master/cargo/flatpak-cargo-generator.py"
  chmod +x "$GENERATOR"
fi

# ── Step 4: generate cargo-sources.json (must redo when Cargo.lock changes) ──
echo "==> Generating cargo-sources.json from Cargo.lock..."
python3 "$GENERATOR" "$REPO_ROOT/Cargo.lock" -o "$FLATPAK_DIR/cargo-sources.json"

# ── Step 5: build & install locally ──────────────────────────
echo "==> Building Flatpak (this takes a while the first time)..."
cd "$REPO_ROOT"
flatpak-builder \
  --user \
  --install \
  --force-clean \
  "$BUILD_DIR" \
  "$FLATPAK_DIR/io.github.paintfe.PaintFE.yml"

echo ""
echo "Done! Run with:"
echo "  flatpak run io.github.paintfe.PaintFE"
