//! Bounded, same-user Windows transport between the MCP bridge and GUI.
use std::{
    ffi::c_void,
    fs::OpenOptions,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use serde_json::Value;

const REQUEST_LIMIT: usize = 512 * 1024;
const RESPONSE_LIMIT: usize = 4 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(15);
const CLIENTS: usize = 4;
const PIPE_NOWAIT: u32 = 1;
const INVALID_HANDLE: isize = -1;
const PIPE_NAME: &str = r"\\.\pipe\PaintFE_Automation_";

#[repr(C)]
struct SecurityAttributes {
    length: u32,
    descriptor: *mut c_void,
    inherit: i32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateNamedPipeW(
        name: *const u16,
        access: u32,
        mode: u32,
        instances: u32,
        output: u32,
        input: u32,
        timeout: u32,
        security: *const c_void,
    ) -> isize;
    fn ConnectNamedPipe(pipe: isize, overlapped: *const c_void) -> i32;
    fn DisconnectNamedPipe(pipe: isize) -> i32;
    fn SetNamedPipeHandleState(
        pipe: RawHandle,
        mode: *const u32,
        max_count: *const u32,
        timeout: *const u32,
    ) -> i32;
    fn ReadFile(
        file: isize,
        buffer: *mut u8,
        len: u32,
        count: *mut u32,
        overlapped: *const c_void,
    ) -> i32;
    fn WriteFile(
        file: isize,
        buffer: *const u8,
        len: u32,
        count: *mut u32,
        overlapped: *const c_void,
    ) -> i32;
    fn LocalFree(memory: *mut c_void) -> *mut c_void;
}

#[link(name = "advapi32")]
unsafe extern "system" {
    fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
        text: *const u16,
        revision: u32,
        descriptor: *mut *mut c_void,
        size: *mut u32,
    ) -> i32;
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

fn os_error() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

pub(crate) struct Request {
    pub envelope: Value,
    pub reply: mpsc::SyncSender<Value>,
}

pub(crate) struct Server {
    pub endpoint: String,
    pub incoming: mpsc::Receiver<Request>,
    stop: Arc<AtomicBool>,
}

impl Server {
    pub(crate) fn start() -> Result<Self, String> {
        let sid = super::owner_sid()?;
        let endpoint = uuid::Uuid::new_v4().to_string();
        let name = wide(&format!("{PIPE_NAME}{endpoint}"));
        let sddl = wide(&format!("D:P(A;;GA;;;{sid})"));
        let mut descriptor = std::ptr::null_mut();
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err("Could not create the local automation access rules".into());
        }
        let security = SecurityAttributes {
            length: std::mem::size_of::<SecurityAttributes>() as u32,
            descriptor,
            inherit: 0,
        };
        let handles: Result<Vec<_>, _> = (0..CLIENTS)
            .map(|index| {
                let first_instance = if index == 0 { 0x0008_0000 } else { 0 };
                let handle = unsafe {
                    CreateNamedPipeW(
                        name.as_ptr(),
                        3 | first_instance,
                        PIPE_NOWAIT | 8,
                        CLIENTS as u32,
                        65_536,
                        65_536,
                        0,
                        std::ptr::from_ref(&security).cast(),
                    )
                };
                if handle == INVALID_HANDLE {
                    Err(format!(
                        "Could not start the local automation pipe ({})",
                        os_error()
                    ))
                } else {
                    Ok(unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) })
                }
            })
            .collect();
        unsafe {
            LocalFree(descriptor);
        }
        let (sender, incoming) = mpsc::sync_channel(CLIENTS);
        let server = Self {
            endpoint,
            incoming,
            stop: Arc::new(AtomicBool::new(false)),
        };
        for pipe in handles? {
            let sender = sender.clone();
            let stop = server.stop.clone();
            std::thread::Builder::new()
                .name("automation-pipe".into())
                .spawn(move || serve(pipe, sender, stop))
                .map_err(|e| e.to_string())?;
        }
        Ok(server)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn serve(pipe: OwnedHandle, sender: mpsc::SyncSender<Request>, stop: Arc<AtomicBool>) {
    let handle = pipe.as_raw_handle();
    while !stop.load(Ordering::Relaxed) {
        if unsafe { ConnectNamedPipe(handle as isize, std::ptr::null()) } == 0 && os_error() == 535
        {
            serve_client(handle, &sender, &stop);
            unsafe {
                DisconnectNamedPipe(handle as isize);
            }
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn serve_client(handle: RawHandle, sender: &mpsc::SyncSender<Request>, stop: &AtomicBool) {
    // Connections are reused. Close idle ones to free a slot for another client.
    while let Ok(bytes) = read_frame(handle, REQUEST_LIMIT, Instant::now() + TIMEOUT, stop) {
        let Ok(envelope) = serde_json::from_slice(&bytes) else {
            break;
        };
        let (reply, response) = mpsc::sync_channel(1);
        if sender.try_send(Request { envelope, reply }).is_err() {
            break;
        }
        let deadline = Instant::now() + TIMEOUT;
        let value = loop {
            if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                return;
            }
            match response.recv_timeout(Duration::from_millis(10)) {
                Ok(value) => break value,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        };
        let Ok(response) = serde_json::to_vec(&value) else {
            break;
        };
        if write_frame(handle, &response, RESPONSE_LIMIT, deadline, stop).is_err() {
            break;
        }
    }
}

pub(super) struct Client {
    pipe: std::fs::File,
    token: String,
    last_used: Instant,
}

impl Client {
    pub(super) fn connect(session: &Value) -> Result<Self, String> {
        let endpoint = session["pipe"]
            .as_str()
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
            .ok_or("Invalid automation endpoint")?;
        let token = session["token"]
            .as_str()
            .ok_or("Invalid automation session")?
            .to_owned();
        let pipe = OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!("{PIPE_NAME}{endpoint}"))
            .map_err(|e| format!("Could not connect to local PaintFE automation: {e}"))?;
        if unsafe {
            SetNamedPipeHandleState(
                pipe.as_raw_handle(),
                &PIPE_NOWAIT,
                std::ptr::null(),
                std::ptr::null(),
            )
        } == 0
        {
            return Err("Could not configure the local automation pipe".into());
        }
        Ok(Self {
            pipe,
            token,
            last_used: Instant::now(),
        })
    }

    pub(super) fn reusable(&self) -> bool {
        self.last_used.elapsed() < Duration::from_secs(10)
    }

    pub(super) fn call(&mut self, params: &Value) -> Result<Value, String> {
        let bytes =
            serde_json::to_vec(&super::envelope(&self.token, params)).map_err(|e| e.to_string())?;
        let handle = self.pipe.as_raw_handle();
        let deadline = Instant::now() + TIMEOUT;
        let stop = AtomicBool::new(false);
        write_frame(handle, &bytes, REQUEST_LIMIT, deadline, &stop)?;
        let response = read_frame(handle, RESPONSE_LIMIT, deadline, &stop)?;
        self.last_used = Instant::now();
        serde_json::from_slice(&response).map_err(|e| e.to_string())
    }
}

fn read_frame(
    handle: RawHandle,
    max_size: usize,
    deadline: Instant,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    let mut header = [0u8; 4];
    transfer(&mut header, deadline, stop, |part, count| unsafe {
        ReadFile(
            handle as isize,
            part.as_mut_ptr(),
            part.len() as u32,
            count,
            std::ptr::null(),
        )
    })?;
    let size = u32::from_le_bytes(header) as usize;
    if size == 0 || size > max_size {
        return Err("Invalid automation frame size".into());
    }
    let mut data = vec![0; size];
    transfer(&mut data, deadline, stop, |part, count| unsafe {
        ReadFile(
            handle as isize,
            part.as_mut_ptr(),
            part.len() as u32,
            count,
            std::ptr::null(),
        )
    })?;
    Ok(data)
}

fn write_frame(
    handle: RawHandle,
    data: &[u8],
    max_size: usize,
    deadline: Instant,
    stop: &AtomicBool,
) -> Result<(), String> {
    if data.is_empty() || data.len() > max_size {
        return Err("Invalid automation frame size".into());
    }
    let mut frame = Vec::with_capacity(4 + data.len());
    frame.extend((data.len() as u32).to_le_bytes());
    frame.extend(data);
    transfer(&mut frame, deadline, stop, |part, count| unsafe {
        WriteFile(
            handle as isize,
            part.as_ptr(),
            part.len() as u32,
            count,
            std::ptr::null(),
        )
    })
}

fn transfer(
    mut data: &mut [u8],
    deadline: Instant,
    stop: &AtomicBool,
    mut io: impl FnMut(&mut [u8], &mut u32) -> i32,
) -> Result<(), String> {
    while !data.is_empty() {
        if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
            return Err("Local automation timed out or stopped".into());
        }
        let mut count = 0;
        if io(data, &mut count) == 0 && os_error() != 232 {
            return Err("Local PaintFE automation connection closed".into());
        }
        if count == 0 {
            std::thread::sleep(Duration::from_millis(5));
        } else {
            data = &mut data[count as usize..];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn two_clients_reuse_bounded_connections_and_preserve_responses() {
        roundtrip(50);
    }

    #[test]
    #[ignore = "Opt-in Windows automation transport performance check"]
    fn ten_thousand_pipe_requests() {
        roundtrip(500);
        roundtrip(5000);
    }

    fn roundtrip(per_client: usize) {
        let start = Instant::now();
        let server = Server::start().unwrap();
        let session = json!({"pipe":server.endpoint,"token":"test-session"});
        let p95 = std::thread::scope(|scope| {
            let clients = (0..2)
                .map(|client_id| {
                    let session = &session;
                    scope.spawn(move || {
                        let mut client = Client::connect(session).unwrap();
                        let mut timings = Vec::with_capacity(per_client);
                        for sequence in 0..per_client {
                            let params = json!({"client":client_id,"sequence":sequence});
                            let sent = Instant::now();
                            assert_eq!(client.call(&params).unwrap(), params);
                            timings.push(sent.elapsed());
                        }
                        assert!(
                            client
                                .call(&json!({"oversize":"x".repeat(REQUEST_LIMIT)}))
                                .is_err()
                        );
                        timings
                    })
                })
                .collect::<Vec<_>>();
            for _ in 0..per_client * 2 {
                let request = server.incoming.recv_timeout(TIMEOUT).unwrap();
                assert_eq!(request.envelope["token"], "test-session");
                request
                    .reply
                    .send(request.envelope["params"].clone())
                    .unwrap();
            }
            let mut all = clients
                .into_iter()
                .flat_map(|client| client.join().unwrap())
                .collect::<Vec<_>>();
            all.sort_unstable();
            all[all.len() * 95 / 100]
        });
        eprintln!(
            "{} requests, elapsed {:?}, p95 {:?}",
            2 * per_client,
            start.elapsed(),
            p95
        );
    }
}
