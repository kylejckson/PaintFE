# build-msix.ps1 — Build a PaintFE MSIX package locally (for testing / submission)
# Run from the repo root: powershell -File packaging\msix\build-msix.ps1
#
# Prerequisites:
#   - Windows SDK (makeappx.exe / signtool.exe) — install via Visual Studio Installer:
#       Individual Components → "Windows 10 SDK" or "Windows 11 SDK"
#   - Rust toolchain (cargo)
# ──────────────────────────────────────────────────────────────────────────────

param(
    [string]$Version     = "1.0.0.0",    # Must be Major.Minor.Patch.0 format
    [switch]$SkipBuild,                   # Skip cargo build (use existing binary)
    [switch]$SkipAssets                   # Skip icon asset generation
)

$ErrorActionPreference = "Stop"
$repo    = $PSScriptRoot | Split-Path -Parent | Split-Path -Parent
$msixDir = Join-Path $repo "packaging\msix"
$layout  = Join-Path $repo "packaging\msix\PackageLayout"
$output  = Join-Path $repo "PaintFE.msix"

Set-Location $repo

# ── Step 1: Build release binary ─────────────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host "==> [1/5] Building release binary..."
    cargo build --release
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo build failed"; exit 1 }
} else {
    Write-Host "==> [1/5] Skipping build (--SkipBuild)"
}

$bin = Join-Path $repo "target\release\PaintFE.exe"
if (-not (Test-Path $bin)) {
    Write-Error "Binary not found at $bin — run without -SkipBuild"
    exit 1
}

# ── Step 2: Generate Store icon assets ───────────────────────────────────────
if (-not $SkipAssets) {
    Write-Host "==> [2/5] Generating Store icon assets..."
    & powershell -File "$msixDir\gen-assets.ps1"
    if ($LASTEXITCODE -ne 0) { Write-Error "gen-assets.ps1 failed"; exit 1 }
} else {
    Write-Host "==> [2/5] Skipping asset generation (--SkipAssets)"
}

# ── Step 3: Stage PackageLayout ──────────────────────────────────────────────
Write-Host "==> [3/5] Staging PackageLayout..."
Remove-Item -Recurse -Force $layout -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $layout | Out-Null

# Binary
Copy-Item $bin "$layout\PaintFE.exe"

# Manifest — inject version number (raw bytes to guarantee BOM-free UTF-8 output)
$bytes = [System.IO.File]::ReadAllBytes("$msixDir\AppxManifest.xml")
if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    $bytes = $bytes[3..($bytes.Length - 1)]
}
$manifest = [System.Text.Encoding]::UTF8.GetString($bytes)
$manifest = $manifest -creplace 'Version="[\d.]+"', "Version=`"$Version`""
$outBytes = [System.Text.Encoding]::UTF8.GetBytes($manifest)
[System.IO.File]::WriteAllBytes("$layout\AppxManifest.xml", $outBytes)

# Store assets
$assetsOut = Join-Path $msixDir "assets"
if (Test-Path $assetsOut) {
    Copy-Item -Recurse -Force $assetsOut "$layout\assets"
} else {
    Write-Warning "No assets folder found at $assetsOut — run without -SkipAssets first"
}

# ── Step 4: Find makeappx.exe ─────────────────────────────────────────────────
Write-Host "==> [4/5] Locating makeappx.exe..."
$sdkPaths = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.22621.0\x64\makeappx.exe",
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.19041.0\x64\makeappx.exe",
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.18362.0\x64\makeappx.exe"
)
# Also search dynamically
$dynamic = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Filter "makeappx.exe" -Recurse -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match "x64" } |
           Sort-Object LastWriteTime -Descending |
           Select-Object -First 1 -ExpandProperty FullName

$makeappx = ($sdkPaths + @($dynamic)) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $makeappx) {
    Write-Error @"
makeappx.exe not found. Install the Windows SDK:
  Open Visual Studio Installer → Modify → Individual Components
  → Search "Windows 10 SDK" → tick the latest version → Install
"@
    exit 1
}
Write-Host "  Using: $makeappx"

# ── Step 5: Pack MSIX ─────────────────────────────────────────────────────────
Write-Host "==> [5/5] Packing MSIX..."
Remove-Item $output -ErrorAction SilentlyContinue
& $makeappx pack /d $layout /p $output /nv
if ($LASTEXITCODE -ne 0) { Write-Error "makeappx pack failed"; exit 1 }

Write-Host ""
Write-Host "================================================"
Write-Host "  MSIX built: $output"
Write-Host "================================================"
Write-Host ""
Write-Host "NOTE: This MSIX has placeholder Publisher info."
Write-Host "      Before uploading to Partner Center, update AppxManifest.xml with:"
Write-Host "        PLACEHOLDER_PUBLISHER_ID  -> your real Publisher string"
Write-Host "        PLACEHOLDER_PACKAGE_NAME  -> your real Package Identity Name"
Write-Host ""
Write-Host "Partner Center will re-sign the package — no code signing cert needed"
Write-Host "for Store submission. For local sideload testing only, see:"
Write-Host "  https://learn.microsoft.com/en-us/windows/msix/package/signing-package-overview"
