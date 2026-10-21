#!/usr/bin/env bash
# INPUT has one directory per final, checksummed release. Output is a Pages site.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$(realpath "${1:?usage: build-repository.sh <packages-dir> <site-dir>}")"
mkdir -p "${2:?site directory required}"
SITE="$(realpath "$2")"
source "$ROOT/packaging/linux/signing-key.sh"
BASE="$SITE/packages"
mkdir -p "$BASE/apt/pool/main" "$BASE/apt/dists/stable/main/binary-amd64" "$BASE/rpm"
# Use versioned repository filenames: cached metadata must never fetch a new
# package at an old URL. GitHub Release download names remain stable.
gpg --batch --armor --export "$SIGNING_KEY" >"$BASE/paintfe.asc"
rpm --import "$BASE/paintfe.asc"
shopt -s nullglob
PACKAGES=("$INPUT"/*/PaintFE-linux-x64.deb)
test "${#PACKAGES[@]}" -gt 0
for DEB in "${PACKAGES[@]}"; do
  RPM="$(dirname "$DEB")/PaintFE-linux-x64.rpm"
  DEB_VERSION="$(dpkg-deb -f "$DEB" Version)"
  RPM_VERSION="$(rpm -qp --qf '%{VERSION}' "$RPM")"
  [[ "$DEB_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
  test "$DEB_VERSION" = "$RPM_VERSION"
  rpmkeys --checksig "$RPM" | grep -q 'digests signatures OK'
  cp "$DEB" "$BASE/apt/pool/main/paintfe_${DEB_VERSION}_amd64.deb"
  cp "$RPM" "$BASE/rpm/paintfe-${RPM_VERSION}-1.x86_64.rpm"
done
(
  cd "$BASE/apt"
  apt-ftparchive packages pool >dists/stable/main/binary-amd64/Packages
  gzip -n -k dists/stable/main/binary-amd64/Packages
  apt-ftparchive -o APT::FTPArchive::Release::Origin=PaintFE \
    -o APT::FTPArchive::Release::Label=PaintFE \
    -o APT::FTPArchive::Release::Suite=stable \
    -o APT::FTPArchive::Release::Codename=stable \
    -o APT::FTPArchive::Release::Architectures=amd64 \
    -o APT::FTPArchive::Release::Components=main \
    release dists/stable >dists/stable/Release
  sign_file --clearsign --output dists/stable/InRelease dists/stable/Release
  sign_file --armor --detach-sign --output dists/stable/Release.gpg dists/stable/Release
  gpg --verify dists/stable/InRelease
)
createrepo_c "$BASE/rpm"
sign_file --armor --detach-sign --output "$BASE/rpm/repodata/repomd.xml.asc" "$BASE/rpm/repodata/repomd.xml"
gpg --verify "$BASE/rpm/repodata/repomd.xml.asc" "$BASE/rpm/repodata/repomd.xml"
cp "$ROOT/packaging/linux/index.html" "$BASE/index.html"
touch "$SITE/.nojekyll"
# No silent oversized Pages uploads. This is unrelated to Actions retention.
test "$(du -sm "$SITE" | cut -f1)" -lt 900
