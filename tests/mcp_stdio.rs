//! Real executable transport checks, independent of a desktop session.
#![cfg(not(target_arch = "wasm32"))]
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::Duration,
};

struct Bridge {
    child: Child,
    messages: mpsc::Receiver<Value>,
    root: PathBuf,
}
impl Bridge {
    fn start() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/mcp-stdio-tests")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&root).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_PaintFE"))
            .arg("--mcp")
            .env("PAINTFE_TEST_STORAGE_DIR", &root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, messages) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let value =
                    serde_json::from_str(&line).expect("stdout must contain only MCP JSON lines");
                if sender.send(value).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            messages,
            root,
        }
    }
    fn send(&mut self, value: Value) {
        let input = self.child.stdin.as_mut().unwrap();
        writeln!(input, "{value}").unwrap();
        input.flush().unwrap();
    }
    fn read(&self) -> Value {
        self.messages
            .recv_timeout(Duration::from_secs(10))
            .expect("MCP response must arrive without closing stdin or sending a blank line")
    }
}
impl Drop for Bridge {
    fn drop(&mut self) {
        self.child.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn handshake_and_tools_work_without_gui_or_permissions() {
    let mut bridge = Bridge::start();
    bridge.send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"transport regression","version":"1"}}}));
    let init = bridge.read();
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(init["result"]["serverInfo"]["name"], "PaintFE");
    bridge.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    bridge.send(json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}));
    let reply = bridge.read();
    assert_eq!(reply["id"], 2, "notifications must not produce replies");
    let tools = reply["result"]["tools"].as_array().unwrap();
    for name in ["documents", "stroke", "activate_layer", "job_status"] {
        assert!(tools.iter().any(|tool| tool["name"] == name));
    }
    let activate = tools
        .iter()
        .find(|tool| tool["name"] == "activate_layer")
        .unwrap();
    assert!(
        activate["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("revision"))
    );
    bridge.send(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"documents","arguments":{}}}));
    let reply = bridge.read();
    assert_eq!(reply["id"], 3);
    assert!(
        reply["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Open PaintFE")
    );
    assert!(!bridge.root.join("automation/session.json").exists());
}

#[test]
fn unicode_and_escaped_newlines_remain_one_message_and_errors_do_not_break_next_request() {
    let mut bridge = Bridge::start();
    bridge.send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}));
    assert_eq!(bridge.read()["result"]["protocolVersion"], "2025-11-25");
    bridge.send(json!({"jsonrpc":"2.0","id":"тест\n日本語","method":"ping"}));
    assert_eq!(
        bridge.read(),
        json!({"jsonrpc":"2.0","id":"тест\n日本語","result":{}})
    );
    bridge.send(json!({"jsonrpc":"2.0","id":2,"method":"not_a_method"}));
    assert!(bridge.read().get("error").is_some());
    bridge.send(json!({"jsonrpc":"2.0","id":3,"method":"ping"}));
    assert_eq!(bridge.read()["result"], json!({}));
}
