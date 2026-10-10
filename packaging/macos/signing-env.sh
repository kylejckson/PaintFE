#!/usr/bin/env bash
# Source in an isolated CI job. Keep secret values out of command tracing/logs.
set -euo pipefail
if [[ -n "${MACOS_SIGNING_IDENTITY:-}" ]]; then
  : "${MACOS_CERTIFICATE_BASE64:?Signing certificate is required}"
  : "${MACOS_CERTIFICATE_PASSWORD:?Certificate password is required}"
  : "${MACOS_APPLE_ID:?Apple notarization account is required}"
  : "${MACOS_APP_PASSWORD:?Notarization app password is required}"
  : "${MACOS_TEAM_ID:?Apple team is required}"
  cert_path="$RUNNER_TEMP/paintfe-signing.p12"
  keychain_path="$RUNNER_TEMP/paintfe-signing.keychain-db"
  keychain_password=$(openssl rand -hex 24)
  printf '%s' "$MACOS_CERTIFICATE_BASE64" | base64 --decode > "$cert_path"
  chmod 600 "$cert_path"
  security create-keychain -p "$keychain_password" "$keychain_path"
  security set-keychain-settings -lut 21600 "$keychain_path"
  security unlock-keychain -p "$keychain_password" "$keychain_path"
  security import "$cert_path" -P "$MACOS_CERTIFICATE_PASSWORD" -k "$keychain_path" -T /usr/bin/codesign
  security set-key-partition-list -S apple-tool:,apple: -k "$keychain_password" "$keychain_path" >/dev/null
  security list-keychains -d user -s "$keychain_path" "$HOME/Library/Keychains/login.keychain-db"
  rm -f "$cert_path"
  export MACOS_NOTARY_PROFILE=paintfe-release
  export MACOS_NOTARY_KEYCHAIN="$keychain_path"
  xcrun notarytool store-credentials "$MACOS_NOTARY_PROFILE" --apple-id "$MACOS_APPLE_ID" --password "$MACOS_APP_PASSWORD" --team-id "$MACOS_TEAM_ID" --keychain "$keychain_path"
  # Both the identity and notary credentials are confined to this temporary keychain.
  trap 'security delete-keychain "$keychain_path" || true' EXIT
fi
