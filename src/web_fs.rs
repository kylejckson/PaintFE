// Browser-only stand-in for `std::fs::File`, used ONLY on wasm32 builds.
//
// PaintFE's save/export code writes via `File::create(path)` wrapped in a
// `BufWriter`, then relies on the writer being dropped to finish the file.
// This type mimics that exact shape: `write`/`flush` buffer bytes in memory,
// and on `Drop` the buffered bytes are handed to the browser as a download
// (Blob + temporary <a download> click), using the path's file name as the
// suggested download name. `open` always fails — there is no real
// filesystem to read from in the browser; web reads instead go through
// byte-oriented paths (drag & drop, <input type=file>).
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub struct File {
    buf: Vec<u8>,
    pos: usize,
    filename: String,
    committed: bool,
}

impl File {
    pub fn create(path: &Path) -> io::Result<Self> {
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "download".to_string());
        Ok(Self {
            buf: Vec::new(),
            pos: 0,
            filename,
            committed: false,
        })
    }

    pub fn open(_path: &Path) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no filesystem in the browser — use drag & drop or the file picker instead",
        ))
    }
}

// Some encoders (TIFF, ICO) seek backward to patch header offsets after
// writing data, so this behaves like a real in-memory random-access file
// rather than an append-only buffer.
impl Write for File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let end = self.pos + buf.len();
        if end > self.buf.len() {
            self.buf.resize(end, 0);
        }
        self.buf[self.pos..end].copy_from_slice(buf);
        self.pos = end;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for File {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => self.buf.len() as i64 + p,
            SeekFrom::Current(p) => self.pos as i64 + p,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to a negative position",
            ));
        }
        self.pos = new_pos as usize;
        Ok(self.pos as u64)
    }
}

impl Read for File {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no filesystem in the browser",
        ))
    }
}

impl Drop for File {
    fn drop(&mut self) {
        if self.committed || self.buf.is_empty() {
            return;
        }
        self.committed = true;
        trigger_download(&self.filename, &self.buf);
    }
}

/// Trigger a browser file download of `bytes` named `filename`, via a
/// temporary Blob URL and a synthetic `<a download>` click. Best-effort:
/// failures are swallowed since there's no good recovery on drop.
pub fn trigger_download(filename: &str, bytes: &[u8]) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;

    let result = (|| -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;

        let array = js_sys::Uint8Array::from(bytes);
        let parts = js_sys::Array::new();
        parts.push(&array);
        let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)?;

        let anchor = document
            .create_element("a")?
            .dyn_into::<web_sys::HtmlAnchorElement>()?;
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.style().set_property("display", "none").ok();
        let body = document.body().ok_or("no body")?;
        body.append_child(&anchor)?;
        anchor.click();
        body.remove_child(&anchor)?;
        web_sys::Url::revoke_object_url(&url)?;
        Ok(())
    })();

    if let Err(e) = result {
        web_sys::console::error_2(&"PaintFE web download failed:".into(), &e);
    }
}
