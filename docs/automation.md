# Local AI agent control

PaintFE can act as an MCP tool server for an external agent. No model, account or AI subscription is included. The integration is disabled by default.

Open Preferences > AI and enable local agent connections. Reading is separate from document editing; sandboxed scripts require another permission. Copy MCP connection settings into a client that supports local stdio servers. The command is your existing PaintFE executable with `--mcp`. Keep the editor open. Disabling connections cancels pending edits and removes the live session.

The bridge uses standard MCP JSON-RPC over stdin/stdout. Its connection to the GUI uses bounded request files in an owner-restricted local session directory. It opens no network listener and never fetches an AI model. A connected cloud AI client may upload document previews to its provider, so choose your client accordingly.

Tools list documents/layers and revisions, return bounded previews, paint brush/pencil strokes, draw outlined rectangles/ellipses, select rectangles, clear selections, add/select layers, set layer opacity, undo/redo and run bounded raster scripts. A straight line is a two-point stroke. Coordinates are document pixels, independent of zoom and window position. Alpha is part of the RGBA colour.

Before editing, call `documents` and supply its document, layer and revision identities. Stale revisions are rejected. Raster operations require an 8-bit active raster layer, no mask editing, and at most 16 million pixels. They run away from the UI thread and apply only to their original target. Use `job_status` to wait for completion. Each completed raster job is one undo entry. `cancel` or Disconnect and stop discards a pending result. Manual edits invalidate a pending agent result.

Script processing uses the existing isolated same-binary worker and sandbox. No shell, network, arbitrary filesystem, modules or document resizing is granted. Agent tools cannot save, close or overwrite documents. Use PaintFE's normal save workflow after reviewing the result. Editable text, arbitrary UI interaction and remote HTTP clients are outside this first MCP tool set.

Linux/macOS use the same protocol and a private owner-only session directory. Windows restricts its directory to the current user's SID. No theme or community pack can enable these permissions.

## Connect a client

1. Open the desktop editor, then Preferences > AI. Enable local agent connections. Enable document editing only when you want the agent to make changes, and scripts only when needed.
2. Copy the connection settings shown there into a client that supports local MCP stdio servers. Keep the desktop editor open under the same user account and storage profile.
3. Ask the agent to list documents and inspect a preview before editing. Review its changes in PaintFE and save them yourself.

A common client configuration looks like this. Replace the executable path; client configuration formats can differ.

```json
{
  "mcpServers": {
    "paintfe": {
      "command": "C:\\Apps\\PaintFE\\PaintFE.exe",
      "args": ["--mcp"]
    }
  }
}
```

On Linux/macOS, use the path to the installed PaintFE executable instead. This does not start an HTTP service or require an API key in PaintFE. Your chosen agent client has its own model/provider configuration.

## Tool reference

All editing tools require the current `document`, `layer` and integer `revision` from `documents`. Get fresh identities after each completed change. The server advertises the full JSON schemas through `tools/list`.

| Tool | Additional inputs and behaviour |
| --- | --- |
| `documents` | Lists open documents, layer identities and revisions, without filesystem paths. |
| `preview` | Takes `document`; returns a bounded PNG preview. |
| `job_status` | Reports the current asynchronous operation. |
| `cancel` | Cancels pending work before its result is applied. |
| `stroke` | `points` are image-coordinate `[x,y]` or `[x,y,pressure]` arrays, with pressure 0 to 1. Supply RGBA `colour` and `size` from 1 to 512. Optional `pencil` and `erase` flags select those paths. At most 4,096 points. |
| `shape` | `shape` is `rectangle` or `ellipse`; supply `x`, `y`, positive `width` and `height`, RGBA `colour` and outline `size`. Draws an outline. |
| `script` | Supply sandboxed Rhai `source`, at most 256 KiB. Requires the separate script permission. |
| `activate_layer` | Activates the identified raster layer. |
| `new_layer` | Supply the new layer's `name`, at most 80 characters. |
| `layer_opacity` | Supply `opacity` from 0 to 1. |
| `select_rectangle` | Supply nonnegative integer `x`, `y` and positive integer `width`, `height`. |
| `deselect` | Clears the selection. |
| `undo` | Undoes one committed document change. |
| `redo` | Redoes one committed document change. |

A useful first request is: “Inspect the current document, add a layer, draw a blue rectangle near the centre, then show me the result.” The agent should query current identities, submit each edit, inspect `job_status` for asynchronous work and refresh the document state before its next edit. Keep instructions within the supported tools; this integration does not operate every toolbar tool or simulate arbitrary mouse/keyboard input.

## Troubleshooting and stopping

- If no editor session is found, keep the GUI open, enable connections and check that the client uses the same executable, Windows user and storage profile. Avoid connecting an unrelated installed copy to an isolated review build.
- If editing is denied, review the separate edit/script permissions. A theme cannot grant them.
- If a revision is stale, query `documents` again and reconsider the requested edit. Do not blindly retry against a newer image.
- If a target is unsupported, choose an 8-bit raster layer within the stated size limit. Text layers and mask editing are outside this first tool set.
- Use `cancel` for pending work, or Disconnect and stop in Preferences to end the session. Use normal Undo for changes already committed.

Local transport does not make a cloud client local. A remote AI provider can receive previews requested by its client. PaintFE itself does not upload your documents to that provider.
