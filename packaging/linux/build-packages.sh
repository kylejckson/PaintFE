#!/usr/bin/env bash
# Wrap the existing x86-64 Linux release; no second application build.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$(realpath "${1:?usage: build-packages.sh <linux-artifact-dir> <output-dir>}")"
mkdir -p "${2:?output directory required}"
OUT="$(realpath "$2")"
VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT/Cargo.toml" | head -1 | tr -d '\r')"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[[:alnum:].-]+)?$ ]] || {
  echo 'Unsupported native package version' >&2
  exit 1
}
# Both package managers sort ~prerelease before the matching stable version.
VERSION="${VERSION/-/~}"
file "$INPUT/PaintFE" | grep -q 'ELF 64-bit.*x86-64'
HOST="$INPUT/paintdotnet-host/PaintFE.PaintDotNetHost"
test "$(sha256sum "$HOST" | cut -d' ' -f1)" = "$(tr -d '\r\n' <"$HOST.sha256")"
WORK="$(mktemp -d)"
trap 'rm -rf -- "$WORK"' EXIT
STAGE="$WORK/stage"
install -Dm755 "$INPUT/PaintFE" "$STAGE/usr/lib/paintfe/PaintFE"
mkdir -p "$STAGE/usr/lib/paintfe/paintdotnet-host" "$STAGE/usr/bin"
cp -a "$INPUT/paintdotnet-host/." "$STAGE/usr/lib/paintfe/paintdotnet-host/"
chmod 755 "$STAGE/usr/lib/paintfe/paintdotnet-host/PaintFE.PaintDotNetHost"
ln -s ../lib/paintfe/PaintFE "$STAGE/usr/bin/PaintFE"
ln -s ../lib/paintfe/PaintFE "$STAGE/usr/bin/paintfe"
install -Dm644 "$ROOT/packaging/appimage/PaintFE.AppDir/PaintFE.desktop" "$STAGE/usr/share/applications/PaintFE.desktop"
sed -i 's/^Exec=PaintFE /Exec=paintfe /' "$STAGE/usr/share/applications/PaintFE.desktop"
install -Dm644 "$ROOT/assets/icons/app_icon.png" "$STAGE/usr/share/icons/hicolor/256x256/apps/PaintFE.png"
install -Dm644 "$ROOT/LICENSE.md" "$STAGE/usr/share/licenses/paintfe/LICENSE"
mkdir -p "$STAGE/usr/share/licenses/paintfe/fonts"
cp "$ROOT"/assets/fonts/*OFL.txt "$STAGE/usr/share/licenses/paintfe/fonts/"

mkdir -p "$STAGE/DEBIAN"
cat >"$STAGE/DEBIAN/control" <<EOF
Package: paintfe
Version: $VERSION
Section: graphics
Priority: optional
Architecture: amd64
Maintainer: PaintFE maintainers <kylejckson@users.noreply.github.com>
Homepage: https://github.com/kylejckson/PaintFE
Installed-Size: $(du -sk "$STAGE/usr" | cut -f1)
Depends: libc6 (>= 2.35), libgcc-s1, libstdc++6, libfontconfig1, libfreetype6, libx11-6, libxkbcommon0, libwayland-client0, libgtk-3-0 | libgtk-3-0t64, libssl3 | libssl3t64, libicu70 | libicu72 | libicu74 | libicu76 | libicu78, zlib1g
Recommends: libvulkan1
Description: GPU-accelerated raster image editor
 PaintFE with the bundled Paint.NET compatibility host.
EOF
dpkg-deb --root-owner-group --build "$STAGE" "$OUT/PaintFE-linux-x64.deb"
rm -r "$STAGE/DEBIAN"

mkdir -p "$WORK/rpm/SPECS"
cat >"$WORK/rpm/SPECS/paintfe.spec" <<EOF
Name: paintfe
Version: $VERSION
Release: 1
Summary: GPU-accelerated raster image editor
License: MIT AND OFL-1.1
URL: https://github.com/kylejckson/PaintFE
BuildArch: x86_64
AutoReqProv: no
Requires: glibc >= 2.35, libgcc, libstdc++, fontconfig, freetype, libX11, libxkbcommon, wayland-libs, gtk3, openssl-libs, libicu, zlib
Recommends: vulkan-loader
# Never strip/rewrite the host: PaintFE verifies its adjacent SHA-256 sidecar.
%global __os_install_post %{nil}
%global debug_package %{nil}
%description
PaintFE with the bundled Paint.NET compatibility host.
%install
mkdir -p %{buildroot}
cp -a "$STAGE/." %{buildroot}/
%files
/usr/bin/PaintFE
/usr/bin/paintfe
/usr/lib/paintfe
/usr/share/applications/PaintFE.desktop
/usr/share/icons/hicolor/256x256/apps/PaintFE.png
%license /usr/share/licenses/paintfe
EOF
mkdir -p "$WORK/rpmdb"
rpm --dbpath "$WORK/rpmdb" --initdb
rpmbuild --dbpath "$WORK/rpmdb" --define "_topdir $WORK/rpm" --define '_build_id_links none' -bb "$WORK/rpm/SPECS/paintfe.spec"
cp "$WORK/rpm/RPMS/x86_64/paintfe-$VERSION-1.x86_64.rpm" "$OUT/PaintFE-linux-x64.rpm"
dpkg-deb --info "$OUT/PaintFE-linux-x64.deb"
rpm --dbpath "$WORK/rpmdb" -qp --requires "$OUT/PaintFE-linux-x64.rpm"
