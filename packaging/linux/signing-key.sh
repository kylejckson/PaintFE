#!/usr/bin/env bash
# Source this only in a release job. Never echo secret values.
set -euo pipefail
export LC_ALL=C
: "${PACKAGES_GPG_PRIVATE_KEY:?Set the package-signing Actions secret first}"
: "${PACKAGES_GPG_PASSPHRASE:?Set the package-signing passphrase secret first}"
export GNUPGHOME
GNUPGHOME="$(mktemp -d)"
chmod 700 "$GNUPGHOME"
trap 'gpgconf --kill gpg-agent || true; rm -rf -- "$GNUPGHOME"' EXIT
printf '%s' "$PACKAGES_GPG_PRIVATE_KEY" | gpg --batch --import
SIGNING_KEY="$(gpg --batch --with-colons --list-secret-keys | awk -F: '$1 == "fpr" { print $10; exit }')"
test -n "$SIGNING_KEY"
unset PACKAGES_GPG_PRIVATE_KEY
export SIGNING_KEY

sign_file() {
  # Passphrase goes through stdin, never command-line arguments or a log.
  printf '%s\n' "$PACKAGES_GPG_PASSPHRASE" | gpg --batch --yes --pinentry-mode loopback \
    --passphrase-fd 0 --local-user "$SIGNING_KEY" "$@"
}

sign_rpm() {
  gpg --batch --armor --export "$SIGNING_KEY" >"$GNUPGHOME/public.asc"
  rpm --import "$GNUPGHOME/public.asc"
  # rpmsign invokes gpg itself; prime the agent, then use its cache.
  printf 'package signing\n' >"$GNUPGHOME/probe"
  sign_file --detach-sign "$GNUPGHOME/probe"
  rpmsign --define "_gpg_name $SIGNING_KEY" --define '_gpg_path %{getenv:GNUPGHOME}' \
    --define '__gpg /usr/bin/gpg' --define '_gpg_digest_algo sha256' --addsign "$1"
  rpmkeys --checksig "$1"
  # An unsigned RPM can pass digest checks. Require an actual RSA header signature.
  local signature
  signature="$(rpm -qp --qf '%{RSAHEADER:pgpsig}' "$1")"
  test -n "$signature" && test "$signature" != '(none)'
}
