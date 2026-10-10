Build a real `PaintFE.app` and DMG on macOS:

```sh
bash packaging/macos/package.sh target/release/PaintFE target/pdn-host/osx-arm64 arm64 1.4.1
```

The application and plugin host stay together inside the bundle. Drag PaintFE
to Applications. No command-line exception or Gatekeeper bypass is required for
a correctly signed and notarized release.

For signed releases, provision a Developer ID Application identity in the builder's
keychain and set `MACOS_SIGNING_IDENTITY`. Store notarization credentials using
`xcrun notarytool store-credentials` and set `MACOS_NOTARY_PROFILE` to that profile.
Do not commit certificates, passwords, or API keys. Signing without notarization
fails packaging. Without credentials, builds produce an explicitly unsigned DMG
for development; this does **not** resolve Gatekeeper warnings for public users.

GitHub Actions uses repository variables `MACOS_SIGNING_IDENTITY` and
`MACOS_TEAM_ID`, plus secrets `MACOS_CERTIFICATE_BASE64` (exported P12),
`MACOS_CERTIFICATE_PASSWORD`, `MACOS_APPLE_ID`, and `MACOS_APP_PASSWORD`.
The workflow imports these into a temporary keychain and removes it on exit.

`generate-cask.sh` generates a Homebrew cask using both final DMG checksums.
Publish the reviewed cask in a tap after the corresponding GitHub release exists.
Users can then install with `brew install --cask <tap>/paintfe` and update with
`brew upgrade --cask paintfe`. Native tests must cover both architectures, a fresh
quarantined download, Applications installation, plugin host startup, notarization,
and a Homebrew install/upgrade. These cannot be validated on a Windows builder.
