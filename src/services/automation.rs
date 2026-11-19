//! Local stdio MCP bridge. The GUI owns all document mutations.
//! The session directory is restricted to the owner. Windows uses a local pipe
//! for each live connection, Unix uses the existing request mailbox.

#[cfg(target_os = "windows")]
pub(crate) mod pipe;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, Read, Write},
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
static STATUS: Mutex<String> = Mutex::new(String::new());
static LAST_REQUEST: Mutex<Option<Instant>> = Mutex::new(None);
pub fn note_request() {
    *LAST_REQUEST.lock().unwrap() = Some(Instant::now());
}
pub static CANCEL: AtomicBool = AtomicBool::new(false);
pub fn status() -> String {
    if LAST_REQUEST
        .lock()
        .unwrap()
        .is_some_and(|t| t.elapsed() < Duration::from_secs(30))
    {
        return crate::t!("automation.active");
    }
    STATUS.lock().unwrap().clone()
}
pub fn set_status(s: String) {
    *STATUS.lock().unwrap() = s;
}
pub fn stop() {
    CANCEL.store(true, Ordering::Relaxed);
}
pub fn root() -> PathBuf {
    if let Some(p) = std::env::var_os("PAINTFE_TEST_STORAGE_DIR") {
        return PathBuf::from(p).join("automation");
    }
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    #[cfg(not(target_os = "windows"))]
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|v| PathBuf::from(v).join(".local/share")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("PaintFE").join("automation")
}
pub fn prepare() -> Result<PathBuf, String> {
    let path = root();
    fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("Unsafe automation directory".into());
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err("Automation directory cannot be a reparse point".into());
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let m = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !m.is_dir() || m.file_type().is_symlink() || m.uid() != unsafe { libc::geteuid() } {
            return Err("Unsafe automation directory".into());
        }
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let sid = owner_sid()?;
        let reset = std::process::Command::new("icacls.exe")
            .arg(&path)
            .arg("/reset")
            .creation_flags(0x08000000)
            .output()
            .map_err(|e| e.to_string())?;
        if !reset.status.success() {
            return Err("Could not reset automation directory permissions".into());
        }
        let permission = format!("*{sid}:(OI)(CI)F");
        let result = std::process::Command::new("icacls.exe")
            .arg(&path)
            .args(["/inheritance:r", "/grant:r", &permission])
            .creation_flags(0x08000000)
            .output()
            .map_err(|e| e.to_string())?;
        if !result.status.success() {
            return Err("Could not restrict automation directory access".into());
        }
    }
    Ok(path)
}
#[cfg(target_os = "windows")]
fn owner_sid() -> Result<String, String> {
    use std::os::windows::process::CommandExt;
    let who = std::process::Command::new("whoami.exe")
        .args(["/user", "/fo", "csv", "/nh"])
        .creation_flags(0x08000000)
        .output()
        .map_err(|e| e.to_string())?;
    let identity = String::from_utf8_lossy(&who.stdout);
    let sid = identity
        .trim()
        .split(',')
        .next_back()
        .unwrap_or("")
        .trim_matches('"');
    if !sid.starts_with("S-1-")
        || !sid
            .bytes()
            .all(|b| b.is_ascii_digit() || b == b'S' || b == b'-')
    {
        return Err("Cannot determine automation owner".into());
    }
    Ok(sid.to_owned())
}

pub fn tools() -> Value {
    let target = json!({"type":"object","properties":{"document":{"type":"string"},"layer":{"type":"string"},"revision":{"type":"integer","minimum":0}},"required":["document","layer","revision"],"additionalProperties":false});
    let mut list = vec![
        json!({"name":"documents","description":"List open documents, revisions and layer identities. No filesystem paths are disclosed.","inputSchema":{"type":"object","properties":{},"additionalProperties":false}}),
        json!({"name":"preview","description":"Read a bounded PNG preview of one document. A cloud client may upload this image to its provider.","inputSchema":{"type":"object","properties":{"document":{"type":"string"}},"required":["document"],"additionalProperties":false}}),
        json!({"name":"job_status","description":"Read progress or failure of the current automation operation.","inputSchema":{"type":"object","properties":{},"additionalProperties":false}}),
        json!({"name":"cancel","description":"Cancel the current operation before applying its result.","inputSchema":{"type":"object","properties":{},"additionalProperties":false}}),
    ];
    for (name, description, extra, required) in [
        (
            "stroke",
            "Paint a brush or pencil stroke in image pixels. One undo step. Points are [x,y,pressure] with pressure from 0 to 1.",
            json!({"points":{"type":"array","minItems":1,"maxItems":4096,"items":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":3}},"colour":{"type":"array","items":{"type":"integer","minimum":0,"maximum":255},"minItems":4,"maxItems":4},"size":{"type":"number","minimum":1,"maximum":512},"pencil":{"type":"boolean"},"erase":{"type":"boolean"}}),
            vec!["points", "colour", "size"],
        ),
        (
            "script",
            "Run bounded sandboxed raster Rhai code. No shell, filesystem, network or document resizing.",
            json!({"source":{"type":"string","maxLength":262144}}),
            vec!["source"],
        ),
        (
            "activate_layer",
            "Make a raster layer active. Does not change pixels.",
            json!({}),
            vec![],
        ),
        (
            "layer_opacity",
            "Set layer opacity from zero to one.",
            json!({"opacity":{"type":"number","minimum":0,"maximum":1}}),
            vec!["opacity"],
        ),
        (
            "shape",
            "Draw an outlined rectangle or ellipse with the production brush raster path.",
            json!({"shape":{"enum":["rectangle","ellipse"]},"x":{"type":"number"},"y":{"type":"number"},"width":{"type":"number","exclusiveMinimum":0},"height":{"type":"number","exclusiveMinimum":0},"size":{"type":"number","minimum":1,"maximum":512},"colour":{"type":"array","items":{"type":"integer","minimum":0,"maximum":255},"minItems":4,"maxItems":4}}),
            vec!["shape", "x", "y", "width", "height", "size", "colour"],
        ),
        (
            "undo",
            "Undo one committed document change.",
            json!({}),
            vec![],
        ),
        (
            "redo",
            "Redo one committed document change.",
            json!({}),
            vec![],
        ),
        (
            "new_layer",
            "Add an empty raster layer.",
            json!({"name":{"type":"string","maxLength":80}}),
            vec!["name"],
        ),
        (
            "select_rectangle",
            "Replace the selection with a rectangle in image coordinates.",
            json!({"x":{"type":"integer","minimum":0},"y":{"type":"integer","minimum":0},"width":{"type":"integer","minimum":1},"height":{"type":"integer","minimum":1}}),
            vec!["x", "y", "width", "height"],
        ),
        ("deselect", "Clear the selection.", json!({}), vec![]),
    ] {
        let mut schema = target.clone();
        schema["properties"]
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        schema["required"]
            .as_array_mut()
            .unwrap()
            .extend(required.into_iter().map(|s| json!(s)));
        list.push(json!({"name":name,"description":description,"inputSchema":schema}));
    }
    json!(list)
}
pub fn bridge() -> i32 {
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut output = std::io::stdout().lock();
    let mut ready = false;
    #[cfg(target_os = "windows")]
    let mut pipe: Option<pipe::Client> = None;
    loop {
        let mut bytes = Vec::new();
        let n = match input
            .by_ref()
            .take(1024 * 1024 + 1)
            .read_until(b'\n', &mut bytes)
        {
            Ok(n) => n,
            Err(_) => return 1,
        };
        if n == 0 {
            return 0;
        }
        if bytes.len() > 1024 * 1024 {
            return 1;
        }
        let request: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => {
                let _ = writeln!(
                    output,
                    "{}",
                    json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Invalid JSON"}})
                );
                continue;
            }
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let method = request["method"].as_str().unwrap_or("");
        let result: Result<Value, String> = match method {
            "initialize" => {
                ready = true;
                let requested = request["params"]["protocolVersion"]
                    .as_str()
                    .unwrap_or("2025-11-25");
                let protocol = if ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
                    .contains(&requested)
                {
                    requested
                } else {
                    "2025-11-25"
                };
                Ok(
                    json!({"protocolVersion":protocol,"capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"PaintFE","version":env!("CARGO_PKG_VERSION")},"instructions":"Local editor control. Enable AI automation in PaintFE Preferences. Get document/layer/revision identities before editing. Editing and scripts require separate permission. Inspect job_status after asynchronous edits."}),
                )
            }
            "ping" => Ok(json!({})),
            "tools/list" if ready => Ok(json!({"tools":tools()})),
            #[cfg(target_os = "windows")]
            "tools/call" if ready => {
                let connected = match pipe.take().filter(pipe::Client::reusable) {
                    Some(client) => Ok(client),
                    None => session_file().and_then(|session| pipe::Client::connect(&session)),
                };
                connected.and_then(|mut client| {
                    let result = client.call(&request["params"]);
                    if result.is_ok() {
                        pipe = Some(client);
                    }
                    result
                })
            }
            #[cfg(not(target_os = "windows"))]
            "tools/call" if ready => call(&request["params"]),
            _ => Err("Unknown method or session not initialized".into()),
        };
        let reply = match result {
            Ok(value) => json!({"jsonrpc":"2.0","id":id,"result":value}),
            Err(message) => {
                json!({"jsonrpc":"2.0","id":id,"error":{"code":-32603,"message":message}})
            }
        };
        if writeln!(output, "{reply}")
            .and_then(|_| output.flush())
            .is_err()
        {
            return 1;
        }
    }
}
fn session_file() -> Result<Value, String> {
    serde_json::from_slice(
        &super::plugins::read_limited(&root().join("session.json"), 4096)
            .map_err(|_| "Open PaintFE and enable AI automation first.")?,
    )
    .map_err(|_| "Invalid automation session".into())
}

pub(crate) fn envelope(token: &str, params: &Value) -> Value {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    json!({"token":token,"params":params,"expires":now+15})
}

#[cfg(not(target_os = "windows"))]
fn call(params: &Value) -> Result<Value, String> {
    let path = root();
    let session = session_file()?;
    let id = uuid::Uuid::new_v4().to_string();
    let pending = path.join(format!("{id}.request"));
    let response = path.join(format!("{id}.response"));
    let token = session["token"]
        .as_str()
        .ok_or("Invalid automation session")?;
    let request = envelope(token, params);
    let bytes = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    if bytes.len() > 512 * 1024 {
        return Err("Request too large".into());
    }
    super::plugins::atomic_write(&pending, &bytes)?;
    let start = Instant::now();
    loop {
        if let Ok(bytes) = super::plugins::read_limited(&response, 4 * 1024 * 1024) {
            let _ = fs::remove_file(&response);
            let _ = fs::remove_file(&pending);
            return serde_json::from_slice(&bytes).map_err(|e| e.to_string());
        }
        if start.elapsed() > Duration::from_secs(15) {
            let _ = fs::remove_file(&pending);
            return Err("PaintFE did not respond. Check that automation is enabled and no dialog is blocking the editor.".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
