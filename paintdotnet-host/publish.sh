#!/usr/bin/env bash
set -euo pipefail

RID="${1:?usage: publish.sh <runtime-id> [output-dir]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${2:-$ROOT/target/pdn-host/$RID}"

dotnet publish \
  "$ROOT/paintdotnet-host/src/PaintFE.PaintDotNetHost/PaintFE.PaintDotNetHost.csproj" \
  -c Release -r "$RID" --self-contained true -o "$OUT"
