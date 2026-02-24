# gen-assets.ps1 — Generate all required Microsoft Store icon assets from app_icon.png
# Run from the repo root: powershell -File packaging\msix\gen-assets.ps1
#
# Requires: Windows PowerShell / PowerShell 5.1+ (uses System.Drawing)
# Output:   packaging\msix\assets\*.png (all required Store sizes)
# ──────────────────────────────────────────────────────────────────────────────

param(
    [string]$SourceIcon = "assets\icons\app_icon.png",
    [string]$OutDir     = "packaging\msix\assets"
)

Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$src  = Join-Path $repo $SourceIcon
$out  = Join-Path $repo $OutDir

if (-not (Test-Path $src)) {
    Write-Error "Source icon not found: $src"
    exit 1
}

New-Item -ItemType Directory -Force -Path $out | Out-Null

# ── Required Store asset sizes ──────────────────────────────────────────────
# Name : [width, height]
$assets = @{
    "StoreLogo.png"           = @(50,  50)
    "Square44x44Logo.png"     = @(44,  44)
    "Square150x150Logo.png"   = @(150, 150)
    "Wide310x150Logo.png"     = @(310, 150)
    "SplashScreen.png"        = @(620, 300)
    # Scaled variants the Store validator may also expect
    "Square44x44Logo.targetsize-16.png"  = @(16,  16)
    "Square44x44Logo.targetsize-32.png"  = @(32,  32)
    "Square44x44Logo.targetsize-48.png"  = @(48,  48)
    "Square44x44Logo.targetsize-256.png" = @(256, 256)
}

function Resize-Image {
    param([string]$InPath, [string]$OutPath, [int]$W, [int]$H)

    $orig   = [System.Drawing.Image]::FromFile($InPath)
    $bitmap = New-Object System.Drawing.Bitmap($W, $H)
    $g      = [System.Drawing.Graphics]::FromImage($bitmap)

    $g.InterpolationMode  = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
    $g.SmoothingMode      = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $g.PixelOffsetMode    = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

    # Transparent background for non-splash assets
    $g.Clear([System.Drawing.Color]::Transparent)

    # For SplashScreen, use the dark brand background colour
    if ($OutPath -like "*SplashScreen*") {
        $bg = [System.Drawing.ColorTranslator]::FromHtml("#0d0d14")
        $g.Clear($bg)
        # Centre the icon at 256x256 on the splash
        $iconSize = [Math]::Min(256, [Math]::Min($W, $H) - 40)
        $x = ($W - $iconSize) / 2
        $y = ($H - $iconSize) / 2
        $g.DrawImage($orig, $x, $y, $iconSize, $iconSize)
    } else {
        $g.DrawImage($orig, 0, 0, $W, $H)
    }

    $g.Dispose()
    $orig.Dispose()

    $bitmap.Save($OutPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
}

Write-Host "Generating MS Store assets from: $src"
Write-Host "Output directory: $out"
Write-Host ""

foreach ($name in $assets.Keys | Sort-Object) {
    $size    = $assets[$name]
    $outPath = Join-Path $out $name
    Resize-Image -InPath $src -OutPath $outPath -W $size[0] -H $size[1]
    Write-Host "  $name  ($($size[0])x$($size[1]))"
}

Write-Host ""
Write-Host "Done! $($assets.Count) assets written to $out"
