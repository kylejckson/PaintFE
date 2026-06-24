# PaintFE Paint.NET legacy host

This optional, out-of-process host implements a small clean-room compatibility
profile for classic Paint.NET 3.5 CPU `PropertyBasedEffect` plugins. It does not
contain or redistribute Paint.NET binaries.

Publish a self-contained host next to PaintFE:

```bash
dotnet publish src/PaintFE.PaintDotNetHost/PaintFE.PaintDotNetHost.csproj \
  -c Release -r linux-x64 --self-contained true -o ../target/pdn-host/linux-x64
```

Supported RIDs are `win-x64`, `linux-x64`, `osx-x64`, and `osx-arm64`.
During development, set `PAINTFE_PDN_HOST` to the published executable.

The host executes untrusted third-party code and is crash isolation, not a
security sandbox. PaintFE requires explicit opt-in and per-plugin trust.
