#!/usr/bin/env bash
# Run on a macOS builder. Stable DMG names are consumed by the website.
set -euo pipefail
binary=${1:?binary path}
host=${2:?plugin host directory}
arch=${3:?arm64 or x86_64}
version=${4:?version without v}
case "$arch" in arm64|x86_64) ;; *) echo 'Unsupported architecture' >&2; exit 1;; esac
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
app="$stage/PaintFE.app"
mkdir -p "$app/Contents/MacOS/paintdotnet-host" "$app/Contents/Resources"
cp "$binary" "$app/Contents/MacOS/PaintFE"
cp -R "$host/." "$app/Contents/MacOS/paintdotnet-host/"
chmod +x "$app/Contents/MacOS/PaintFE" "$app/Contents/MacOS/paintdotnet-host/PaintFE.PaintDotNetHost"
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>PaintFE</string>
<key>CFBundleIdentifier</key><string>com.paintfe.PaintFE</string>
<key>CFBundleName</key><string>PaintFE</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$version</string>
<key>CFBundleVersion</key><string>$version</string>
<key>CFBundleIconFile</key><string>PaintFE</string>
<key>NSHighResolutionCapable</key><true/>
<key>LSMinimumSystemVersion</key><string>11.0</string>
</dict></plist>
EOF
plutil -lint "$app/Contents/Info.plist"
mkdir -p "$stage/PaintFE.iconset"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" assets/icons/app_icon.png --out "$stage/PaintFE.iconset/icon_${size}x${size}.png" >/dev/null
  twice=$((size * 2))
  sips -z "$twice" "$twice" assets/icons/app_icon.png --out "$stage/PaintFE.iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$stage/PaintFE.iconset" -o "$app/Contents/Resources/PaintFE.icns"
rm -rf "$stage/PaintFE.iconset"
ln -s /Applications "$stage/Applications"

# Apple credentials must be configured by the maintainer. Never bypass Gatekeeper.
if [[ -n "${MACOS_SIGNING_IDENTITY:-}" ]]; then
  # Sign every embedded Mach-O before the outer bundle (including .NET dylibs).
  while IFS= read -r -d '' file_path; do
    if file -b "$file_path" | grep -q 'Mach-O'; then
      codesign --force --timestamp --options runtime --entitlements packaging/macos/entitlements.plist --sign "$MACOS_SIGNING_IDENTITY" "$file_path"
    fi
  done < <(find "$app/Contents" -type f -print0)
  codesign --force --timestamp --options runtime --entitlements packaging/macos/entitlements.plist --sign "$MACOS_SIGNING_IDENTITY" "$app"
  codesign --verify --deep --strict --verbose=2 "$app"
else
  echo '::warning::Developer ID is not configured; this DMG cannot be claimed Gatekeeper-ready.'
fi
output="PaintFE-macOS-${arch}.dmg"
hdiutil create -volname PaintFE -srcfolder "$stage" -ov -format UDZO "$output"
if [[ -n "${MACOS_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --timestamp --sign "$MACOS_SIGNING_IDENTITY" "$output"
  if [[ -z "${MACOS_NOTARY_PROFILE:-}" ]]; then
    echo 'Signing requires MACOS_NOTARY_PROFILE for notarization' >&2
    exit 1
  fi
  notary_args=(--keychain-profile "$MACOS_NOTARY_PROFILE")
  if [[ -n "${MACOS_NOTARY_KEYCHAIN:-}" ]]; then notary_args+=(--keychain "$MACOS_NOTARY_KEYCHAIN"); fi
  xcrun notarytool submit "$output" "${notary_args[@]}" --wait
  xcrun stapler staple "$output"
  xcrun stapler validate "$output"
  spctl --assess --type open --context context:primary-signature --verbose=2 "$output"
fi
