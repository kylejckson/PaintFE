# Linux packages and updates

RPM signing exports an ASCII-armored public key for compatibility with Ubuntu
22.04's RPM tools, verifies the package and requires a real RSA header signature.
CI runs `bash packaging/linux/test-signing.sh` on Ubuntu with a disposable,
password-protected key. It checks signing, incorrect-password rejection and
damaged-package rejection without using release secrets.

PaintFE's DEB and RPM files wrap the existing x86-64 Linux release binary and
Paint.NET compatibility host. There is no separate application edition and no
application update checker. APT or DNF handles downloads and updates.

## Availability and compatibility

The workflow produces `PaintFE-linux-x64.deb` and `PaintFE-linux-x64.rpm` on the
next stable release containing these changes. Older releases do not gain these
files automatically. ARM Linux is not included.

The packages target Ubuntu 22.04 or newer, compatible Debian systems with
glibc 2.35 or newer, and recent Fedora systems. These are dependency targets,
not a claim of desktop testing on those distributions. Other RPM distributions
are not yet supported. Runtime dependency resolution, launching, GPU rendering,
file dialogs and upgrades need user testing on each distribution.

Download packages from the [latest GitHub Release](https://github.com/kylejckson/PaintFE/releases/latest).
For a one-off install, run `sudo apt install ./PaintFE-linux-x64.deb` or
`sudo dnf install ./PaintFE-linux-x64.rpm` from the download directory.
Downloading a package alone does not subscribe the system to future updates.

## Subscribe to updates

These commands become usable after the first successful package Pages deployment.
The expected base URL is `https://kylejckson.github.io/PaintFE/packages`.
If GitHub Pages uses a custom domain, substitute its actual URL throughout.
Before trusting the downloaded key, check its primary fingerprint against:

```text
5BE4 6520 81B7 1E96 9F3C BE80 3790 3A69 E2C8 11BC
```

Without spaces: `5BE4652081B71E969F3CBE8037903A69E2C811BC`.
HTTPS alone is not an independent fingerprint check.

### Ubuntu / Debian

```sh
curl -fsSLo /tmp/paintfe.asc https://kylejckson.github.io/PaintFE/packages/paintfe.asc
gpg --show-keys --with-fingerprint /tmp/paintfe.asc
# After checking the fingerprint:
sudo install -Dm644 /tmp/paintfe.asc /etc/apt/keyrings/paintfe.asc
sudo tee /etc/apt/sources.list.d/paintfe.sources >/dev/null <<'EOF'
Types: deb
URIs: https://kylejckson.github.io/PaintFE/packages/apt
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/paintfe.asc
EOF
sudo apt update
sudo apt install paintfe
```

Later: `sudo apt update && sudo apt upgrade`. Normal desktop update tools can
also discover PaintFE. Automatic installation depends on the user's updater
configuration; unattended-upgrades may require explicitly allowing this origin.

### Fedora

```sh
curl -fsSLo /tmp/paintfe.asc https://kylejckson.github.io/PaintFE/packages/paintfe.asc
gpg --show-keys --with-fingerprint /tmp/paintfe.asc
# After checking the fingerprint:
sudo rpm --import /tmp/paintfe.asc
sudo tee /etc/yum.repos.d/paintfe.repo >/dev/null <<'EOF'
[paintfe]
name=PaintFE
baseurl=https://kylejckson.github.io/PaintFE/packages/rpm
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://kylejckson.github.io/PaintFE/packages/paintfe.asc
EOF
sudo dnf install paintfe
```

Later: `sudo dnf upgrade --refresh paintfe`. DNF may separately ask to trust the
key for package signatures and repository metadata. Verify its fingerprint.

## Maintainer setup (once, before the next release tag)

No new account, domain or paid signing certificate is required. Do not paste
private keys or passphrases into issues, chat, Git, or workflow logs.

### 1. Create a dedicated signing key on your PC

Gpg4win/GnuPG is already installed on the development PC. Open PowerShell in a
private folder **outside the PaintFE repository**, such as a new folder in your
Documents directory. Run:

```powershell
gpg --quick-generate-key "PaintFE Linux packages" rsa4096 sign 2y
gpg --list-secret-keys --keyid-format long --with-fingerprint "PaintFE Linux packages"
```

Choose and save a strong passphrase when GnuPG prompts. Copy the full fingerprint
of the new key (the long hexadecimal line). Substitute it for `FINGERPRINT`:

```powershell
gpg --armor --output paintfe-private.asc --export-secret-keys FINGERPRINT
gpg --armor --output paintfe-public.asc --export FINGERPRINT
gpg --output paintfe-revocation.asc --gen-revoke FINGERPRINT
```

Save the private key, passphrase and revocation certificate in a secure backup.
The revocation certificate is for emergencies; do not import it now. Set a
reminder to renew the key before its two-year expiry. Keep the same key when
renewing so existing users retain trust.

### 2. Add the two repository secrets

Open PaintFE on GitHub → **Settings → Secrets and variables → Actions → Secrets
→ New repository secret**. Add:

| Name | Paste into the value box |
| --- | --- |
| `PACKAGES_GPG_PRIVATE_KEY` | All text from `paintfe-private.asc`, including BEGIN/END lines |
| `PACKAGES_GPG_PASSPHRASE` | The passphrase you just chose |

These names must match exactly. Use **Secrets**, not Variables. Do not add this
key under your account's SSH/GPG signing-key settings; those are unrelated.
The release package job deliberately fails if either secret is missing.

### 3. Enable Pages only after reviewing the existing site

A repository has one Pages deployment. This workflow publishes a complete site
with `/packages/`; enabling it would replace any other Pages site for this repo.
If Pages already serves something important, merge that site's output into this
deployment before enabling it. The external paintfe.com website is not modified
by these source changes, but check whether its hosting depends on this repo's Pages.

After approval to change GitHub settings:

1. Open **Settings → Pages → Build and deployment → Source → GitHub Actions**.
2. Under **Settings → Secrets and variables → Actions → Variables**, add repository
   variable `PACKAGES_PAGES_ENABLED` with value `true`. This explicit opt-in keeps
   package Pages publication disabled until the site arrangement is confirmed.
3. Check the `github-pages` environment allows the default branch to deploy.
4. Publish the public fingerprint in the installation documentation before launch.

The normal short-lived `GITHUB_TOKEN` handles publication; no personal access
token is needed. Existing Actions retention, caches and cleanup settings are
left unchanged.

### 4. Publish after code/commit approval

Push the approved implementation and follow the normal approved release process.
The release workflow builds/signs RPM packages and publishes both installers and
checksums. The package repository workflow then downloads the final release
assets, verifies checksums/signatures, signs the APT/RPM indexes and deploys Pages.
It uses `workflow_run` because releases created with `GITHUB_TOKEN` do not trigger
another workflow through a `release` event.

The Pages repository contains the latest stable package release and one previous
package release, with versioned filenames. GitHub Releases retains the downloads;
this does not change Actions artifact retention. A failed site build leaves the
previous deployment intact. After fixing a deployment problem, rerun **Publish
Linux package repository** manually; it rebuilds from published releases.
Prereleases are not added to the stable repository. Key rotation requires a
planned trust transition, including previous RPM signatures; do not simply
replace the signing key in secrets.

## Checks and user testing

On Linux, install `rpm`, `file`, `apt-utils`, `createrepo-c` and GnuPG. Build the
ordinary Linux binary/host first, then run:

```sh
bash packaging/linux/build-packages.sh target/release target/native-packages
```

This builds unsigned local packages; release CI signs RPM before publication.
APT authenticates DEB downloads through signed repository metadata rather than
an embedded DEB signature. The signing workflow verifies its generated signatures.

Before announcing broad compatibility, ask testers to report their distribution,
version, desktop and graphics driver, then check:

1. Install using the repository and open PaintFE from the application menu.
2. Open/save an image and try native file dialogs and a Paint.NET plugin.
3. Save preferences, upgrade to the next release, and confirm they remain.
4. Remove/reinstall the package and confirm personal data remains in the profile.
5. Confirm `apt update` / `dnf upgrade --refresh` discovers a newer release without
   bypassing signature checks.

Windows-only validation cannot prove package installation, Linux native-library
compatibility, repository client behavior, or desktop launching.
