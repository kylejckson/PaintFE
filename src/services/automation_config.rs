//! Automation permissions are application settings, never theme data.
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutomationSettings {
    pub enabled: bool,
    pub editing: bool,
    pub scripts: bool,
}
pub fn preferences(ui: &mut egui::Ui, s: &mut AutomationSettings) -> bool {
    let before = s.clone();
    ui.checkbox(&mut s.enabled, crate::t!("automation.enable"));
    ui.add_enabled_ui(s.enabled, |ui| {
        ui.checkbox(&mut s.editing, crate::t!("automation.editing"));
        ui.add_enabled_ui(s.editing, |ui| {
            ui.checkbox(&mut s.scripts, crate::t!("automation.scripts"));
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            if ui.button(crate::t!("automation.copy_config")).clicked()
                && let Ok(exe) = std::env::current_exe()
            {
                let mut server = serde_json::json!({"command":exe,"args":["--mcp"]});
                if let Some(profile) = std::env::var_os("PAINTFE_TEST_STORAGE_DIR") {
                    server["env"] =
                        serde_json::json!({"PAINTFE_TEST_STORAGE_DIR":profile.to_string_lossy()});
                }
                ui.ctx()
                    .copy_text(serde_json::json!({"mcpServers":{"paintfe":server}}).to_string());
            }
            ui.weak(crate::services::automation::status());
            if ui.button(crate::t!("automation.stop")).clicked() {
                crate::services::automation::stop();
                s.enabled = false;
            }
        }
    });
    ui.weak(crate::t!("automation.notice"));
    *s != before
}
