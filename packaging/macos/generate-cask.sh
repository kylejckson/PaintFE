#!/usr/bin/env bash
set -euo pipefail
version=${1:?version without v}
arm_sha=$(shasum -a 256 PaintFE-macOS-arm64.dmg | cut -d ' ' -f 1)
intel_sha=$(shasum -a 256 PaintFE-macOS-x86_64.dmg | cut -d ' ' -f 1)
cat <<EOF
cask "paintfe" do
  version "$version"
  on_arm do
    sha256 "$arm_sha"
    url "https://github.com/kylejckson/PaintFE/releases/download/v#{version}/PaintFE-macOS-arm64.dmg"
  end
  on_intel do
    sha256 "$intel_sha"
    url "https://github.com/kylejckson/PaintFE/releases/download/v#{version}/PaintFE-macOS-x86_64.dmg"
  end
  name "PaintFE"
  desc "Offline image editor"
  homepage "https://paintfe.com/"
  depends_on macos: ">= :big_sur"
  app "PaintFE.app"
end
EOF
