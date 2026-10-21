#!/usr/bin/env bash
# Exercise production signing with a tiny RPM and a disposable encrypted key.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/target"
WORK="$(mktemp -d "$ROOT/target/rpm-signing-test.XXXXXX")"
cleanup() {
  case "$WORK" in "$ROOT"/target/rpm-signing-test.*) rm -rf -- "$WORK" ;; *) exit 1 ;; esac
}
trap cleanup EXIT
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR" "$WORK/rpmdb" "$WORK/rpm/SPECS"
# Keep the fixture key out of the runner's package database.
rpm() { command rpm --dbpath "$WORK/rpmdb" "$@"; }
rpmkeys() { command rpmkeys --dbpath "$WORK/rpmdb" "$@"; }
rpm --initdb
export GNUPGHOME="$WORK/keygen"
mkdir -m700 "$GNUPGHOME"
# This password protects only the temporary test key, never a production key.
export PACKAGES_GPG_PASSPHRASE='paintfe-disposable-ci-key'
printf '%s\n' "$PACKAGES_GPG_PASSPHRASE" | gpg --batch --pinentry-mode loopback \
  --passphrase-fd 0 --quick-generate-key 'PaintFE signing CI fixture' rsa2048 sign 1d
fingerprint="$(gpg --batch --with-colons --list-secret-keys | awk -F: '$1 == "fpr" { print $10; exit }')"
export PACKAGES_GPG_PRIVATE_KEY
PACKAGES_GPG_PRIVATE_KEY="$(printf '%s\n' "$PACKAGES_GPG_PASSPHRASE" | gpg --batch \
  --pinentry-mode loopback --passphrase-fd 0 --armor --export-secret-keys "$fingerprint")"
gpgconf --kill gpg-agent

cat >"$WORK/rpm/SPECS/fixture.spec" <<'EOF'
Name: paintfe-signing-fixture
Version: 1.0
Release: 1
Summary: Disposable PaintFE signing test
License: MIT
BuildArch: noarch
%description
One text file for testing RPM signing.
%install
mkdir -p %{buildroot}/usr/share/paintfe-signing-fixture
printf 'fixture\n' > %{buildroot}/usr/share/paintfe-signing-fixture/example.txt
%files
/usr/share/paintfe-signing-fixture/example.txt
EOF
rpmbuild --define "_topdir $WORK/rpm" -bb "$WORK/rpm/SPECS/fixture.spec"
package="$WORK/rpm/RPMS/noarch/paintfe-signing-fixture-1.0-1.noarch.rpm"
test "$(rpm -qp --qf '%{RSAHEADER:pgpsig}' "$package")" = '(none)'

# Source in a subshell so its isolated GPG cleanup does not replace our trap.
(
  source "$ROOT/packaging/linux/signing-key.sh"
  sign_rpm "$package"
  gpg --verify "$GNUPGHOME/probe.sig" "$GNUPGHOME/probe"
)
rpmkeys --checksig "$package" | grep -q 'digests signatures OK'
signature="$(rpm -qp --qf '%{RSAHEADER:pgpsig}' "$package")"
key_id="${fingerprint: -16}"
[[ "${signature,,}" == *"${key_id,,}"* ]]

# A wrong passphrase must fail before signing, without relying on an agent cache.
if (
  export PACKAGES_GPG_PASSPHRASE='incorrect-test-password'
  source "$ROOT/packaging/linux/signing-key.sh"
  sign_file --detach-sign --output "$WORK/wrong.sig" "$package"
); then
  echo 'Signing unexpectedly accepted the wrong passphrase' >&2
  exit 1
fi

# Corrupt a copy and require verification to reject it.
cp "$package" "$WORK/corrupt.rpm"
python3 - "$WORK/corrupt.rpm" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
data = bytearray(p.read_bytes())
data[-1] ^= 1
p.write_bytes(data)
PY
if rpmkeys --checksig "$WORK/corrupt.rpm"; then
  echo 'Verification unexpectedly accepted a damaged RPM' >&2
  exit 1
fi
echo 'RPM signing: armored import, encrypted key, signature and tamper checks passed.'
