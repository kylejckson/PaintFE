//! Single-instance IPC — ensures only one PaintFE GUI runs at a time.
//!
//! On Windows: uses a named pipe (`\\.\pipe\PaintFE_OpenFile`).
//! - New instance tries to connect as client; if successful, sends file path(s) and exits.
//! - First instance creates the pipe server and listens for incoming file paths.
//!
//! On non-Windows: single-instance is not enforced; files just open in the new instance.

use std::path::PathBuf;
use std::sync::mpsc;

// ============================================================================
// Collect positional file arguments from the command line (all platforms)
// ============================================================================

/// Parse `std::env::args()` for positional file paths (not prefixed with `-`).
/// Called only in GUI mode (after CLI detection has already been checked).
pub fn collect_startup_files() -> Vec<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    let mut files = Vec::new();
    let mut skip_next = false;
    for arg in &args[1..] {
        if skip_next {
            skip_next = false;
            continue;
        }
        // Skip flags and their values (e.g. --input, -i, --script, etc.)
        if arg.starts_with('-') {
            // If the flag takes a value, skip the next arg too
            if arg == "--input"
                || arg == "-i"
                || arg == "--script"
                || arg == "-s"
                || arg == "--output"
                || arg == "-o"
                || arg == "--output-dir"
                || arg == "--format"
                || arg == "-f"
                || arg == "--quality"
                || arg == "-q"
                || arg == "--tiff-compression"
            {
                skip_next = true;
            }
            continue;
        }
        let path = PathBuf::from(arg);
        if path.exists() {
            files.push(path);
        }
    }
    files
}

// ============================================================================
// Windows: Named Pipe single-instance IPC
// ============================================================================

#[cfg(target_os = "windows")]
mod platform {
    use super::*;

    const PIPE_NAME: &str = r"\\.\pipe\PaintFE_OpenFile";

    // Win32 constants
    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const OPEN_EXISTING: u32 = 3;
    const INVALID_HANDLE_VALUE: isize = -1;
    const PIPE_ACCESS_INBOUND: u32 = 0x0000_0001;
    const PIPE_TYPE_BYTE: u32 = 0x0000_0000;
    const PIPE_READMODE_BYTE: u32 = 0x0000_0000;
    const PIPE_WAIT: u32 = 0x0000_0000;
    const ERROR_PIPE_CONNECTED: u32 = 535;

    // Win32 FFI declarations (Rust 2024 edition syntax)
    unsafe extern "system" {
        fn CreateNamedPipeW(
            lpName: *const u16,
            dwOpenMode: u32,
            dwPipeMode: u32,
            nMaxInstances: u32,
            nOutBufferSize: u32,
            nInBufferSize: u32,
            nDefaultTimeOut: u32,
            lpSecurityAttributes: *const std::ffi::c_void,
        ) -> isize;

        fn ConnectNamedPipe(
            hNamedPipe: isize,
            lpOverlapped: *const std::ffi::c_void,
        ) -> i32;

        fn DisconnectNamedPipe(hNamedPipe: isize) -> i32;
        fn CloseHandle(hObject: isize) -> i32;

        fn ReadFile(
            hFile: isize,
            lpBuffer: *mut u8,
            nNumberOfBytesToRead: u32,
            lpNumberOfBytesRead: *mut u32,
            lpOverlapped: *const std::ffi::c_void,
        ) -> i32;

        fn WriteFile(
            hFile: isize,
            lpBuffer: *const u8,
            nNumberOfBytesToWrite: u32,
            lpNumberOfBytesWritten: *mut u32,
            lpOverlapped: *const std::ffi::c_void,
        ) -> i32;

        fn CreateFileW(
            lpFileName: *const u16,
            dwDesiredAccess: u32,
            dwShareMode: u32,
            lpSecurityAttributes: *const std::ffi::c_void,
            dwCreationDisposition: u32,
            dwFlagsAndAttributes: u32,
            hTemplateFile: isize,
        ) -> isize;

        fn GetLastError() -> u32;
    }

    /// Encode a Rust string as a null-terminated wide (UTF-16) string for Win32 APIs.
    fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Try to send file paths to an already-running PaintFE instance via named pipe.
    /// Returns `true` if paths were sent successfully (caller should exit).
    pub fn try_send_to_existing(paths: &[PathBuf]) -> bool {
        if paths.is_empty() {
            return false;
        }

        let pipe_name_wide = to_wide(PIPE_NAME);

        unsafe {
            let handle = CreateFileW(
                pipe_name_wide.as_ptr(),
                GENERIC_WRITE,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                0,
            );

            if handle == INVALID_HANDLE_VALUE {
                return false;
            }

            // Serialize paths as newline-separated UTF-8
            let data: String = paths
                .iter()
                .filter_map(|p| p.to_str())
                .collect::<Vec<_>>()
                .join("\n");

            let bytes = data.as_bytes();
            let mut written = 0u32;
            let _ok = WriteFile(
                handle,
                bytes.as_ptr(),
                bytes.len() as u32,
                &mut written,
                std::ptr::null(),
            );

            CloseHandle(handle);
            true
        }
    }

    /// Start the named pipe listener in a background thread.
    /// Received file paths are sent through the returned channel.
    pub fn start_listener() -> mpsc::Receiver<PathBuf> {
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let pipe_name_wide = to_wide(PIPE_NAME);

            loop {
                // Create a new pipe instance for each client connection
                let pipe = unsafe {
                    CreateNamedPipeW(
                        pipe_name_wide.as_ptr(),
                        PIPE_ACCESS_INBOUND,
                        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                        1,    // max instances — one at a time
                        0,    // out buffer size
                        4096, // in buffer size
                        0,    // default timeout
                        std::ptr::null(),
                    )
                };

                if pipe == INVALID_HANDLE_VALUE {
                    // Pipe creation failed — give up
                    break;
                }

                // Block until a client connects
                let connected = unsafe { ConnectNamedPipe(pipe, std::ptr::null()) };
                let last_err = unsafe { GetLastError() };
                let is_connected = connected != 0 || last_err == ERROR_PIPE_CONNECTED;

                if is_connected {
                    // Read the data the client sent
                    let mut buf = vec![0u8; 32768];
                    let mut total_read = 0usize;

                    loop {
                        let mut bytes_read = 0u32;
                        let ok = unsafe {
                            ReadFile(
                                pipe,
                                buf[total_read..].as_mut_ptr(),
                                (buf.len() - total_read) as u32,
                                &mut bytes_read,
                                std::ptr::null(),
                            )
                        };
                        if ok == 0 || bytes_read == 0 {
                            break;
                        }
                        total_read += bytes_read as usize;
                        if total_read >= buf.len() {
                            break;
                        }
                    }

                    if total_read > 0
                        && let Ok(data) = std::str::from_utf8(&buf[..total_read])
                    {
                        for line in data.lines() {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                let path = PathBuf::from(trimmed);
                                if tx.send(path).is_err() {
                                    // Receiver dropped — app is shutting down
                                    unsafe {
                                        DisconnectNamedPipe(pipe);
                                        CloseHandle(pipe);
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }

                unsafe {
                    DisconnectNamedPipe(pipe);
                    CloseHandle(pipe);
                }

                // If the receiver has been dropped, stop listening
                // (We detect this on the next send above, but also
                // do a cheap dummy check here for fast exit.)
            }
        });

        rx
    }

    /// Try to bring the existing PaintFE window to the foreground after
    /// sending files via pipe. Uses `FindWindowW` + `SetForegroundWindow`.
    pub fn focus_existing_window() {
        unsafe extern "system" {
            fn FindWindowW(
                lpClassName: *const u16,
                lpWindowName: *const u16,
            ) -> isize;
            fn SetForegroundWindow(hWnd: isize) -> i32;
            fn ShowWindow(hWnd: isize, nCmdShow: i32) -> i32;
            fn IsIconic(hWnd: isize) -> i32;
        }
        const SW_RESTORE: i32 = 9;

        let title = to_wide("PaintFE");
        unsafe {
            let hwnd = FindWindowW(std::ptr::null(), title.as_ptr());
            if hwnd != 0 {
                // If minimized, restore first
                if IsIconic(hwnd) != 0 {
                    ShowWindow(hwnd, SW_RESTORE);
                }
                SetForegroundWindow(hwnd);
            }
        }
    }
}

// ============================================================================
// Non-Windows: stubs (single-instance not enforced)
// ============================================================================

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::*;

    pub fn try_send_to_existing(_paths: &[PathBuf]) -> bool {
        false
    }

    pub fn start_listener() -> mpsc::Receiver<PathBuf> {
        let (_tx, rx) = mpsc::channel();
        rx
    }

    pub fn focus_existing_window() {}
}

// ============================================================================
// Public API — delegates to platform-specific implementation
// ============================================================================

/// Try to send file paths to an already-running PaintFE instance.
/// Returns `true` if paths were sent (caller should exit the process).
pub fn try_send_to_existing(paths: &[PathBuf]) -> bool {
    platform::try_send_to_existing(paths)
}

/// Start the single-instance listener. Returns a channel receiver that
/// delivers file paths sent from newly-launched PaintFE instances.
pub fn start_listener() -> mpsc::Receiver<PathBuf> {
    platform::start_listener()
}

/// Attempt to bring the existing PaintFE window to the foreground.
pub fn focus_existing_window() {
    platform::focus_existing_window();
}
