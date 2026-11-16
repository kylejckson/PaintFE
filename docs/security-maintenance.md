# Security maintenance

## Release and AUR trust boundary

The release workflow calls AUR publishing only after the public downloads and
optional packs have been verified. AUR publishing is a reusable workflow with a
manual retry option, not a privileged `workflow_run` follower. It checks out
trusted packaging tools from main without persisting Git credentials.

The preparation tool rejects foreign repositories, unexpected event types,
non-numeric versions, unpublished releases, missing Linux downloads, moved tags
and commits outside main. Version strings are data passed through environment
variables, never inserted into shell code. The AUR deploy key is supplied only
to the final publishing action. A manual rerun can target an older published
release which is still an ancestor of main.

## Python build dependencies

All CI and release pip installs use `.github/requirements-build.txt`, exact
versions, `--require-hashes` and `--only-binary=:all:`. The lock includes
cryptography and its transitive dependencies, with hashes for supported wheels.
Update the full lock from reviewed upstream releases, then test wheel selection
on Windows, Linux and both macOS architectures before publishing. Native codec
build tools use the separate `.github/requirements-native.txt` wheel lock, and
their installer explicitly removes the optional-pack signing key. Separating
credential scope does not make an unverified imported dependency safe.

## Fuzzing

See [the fuzzing guide](../fuzz/README.md). Production HEIF metadata and numeric
expression code is exercised without loading the editor. A short mutation smoke
run is part of ordinary CI, and ClusterFuzzLite provides coverage-guided,
AddressSanitizer-backed runs on changes and a weekly schedule. These are bounded
checks, not a claim that all file formats or native libraries have been fuzzed.

## Code-scanning triage

Keep real cryptographic and workflow checks enabled. egui widget ID salts and
the shared numeric editor's ID salts are deterministic UI identities, not
password salts. Verified alerts for those APIs should be classified individually
as false positives, with their actual call path recorded. Do not randomise UI
IDs or disable the whole Rust security suite to clear the dashboard.

The skcms `init_tables` widening warning also has checked call-site bounds:
output channels are at most four and byte width is a literal one or two. The
reported product is at most eight. Recheck this conclusion after vendor updates.

After publication, wait for fresh scanning, verify that genuine findings are
resolved, and only then dismiss the reviewed false positives. Do not dismiss new
or changed findings merely to reach zero. Security alert triage is separate from
ordinary user issues, which remain open for confirmation.
