#[derive(Clone)]
pub struct TiledImage {
    pub width: u32,
    pub height: u32,
    chunks_per_row: u32,
    chunks: Vec<Option<Arc<RgbaImage>>>,
}

impl TiledImage {
    // ---- construction -------------------------------------------------------

    /// Create an empty (fully transparent) tiled image.
    pub fn new(width: u32, height: u32) -> Self {
        // Sanity: clamp dimensions to prevent overflow (max ~256 megapixels)
        let (width, height) = {
            let total = (width as u64) * (height as u64);
            if total > 256_000_000 || width == 0 || height == 0 {
                eprintln!(
                    "TiledImage::new: dimensions {}×{} exceed 256M pixels, clamped to 1×1",
                    width, height
                );
                (1, 1)
            } else {
                (width, height)
            }
        };
        let chunks_per_row = width.div_ceil(CHUNK_SIZE);
        let chunks_per_col = height.div_ceil(CHUNK_SIZE);
        let total = (chunks_per_row * chunks_per_col) as usize;
        Self {
            width,
            height,
            chunks_per_row,
            chunks: vec![None; total],
        }
    }

    /// Fill the entire image with `color`.  Chunks with `alpha == 0` are
    /// skipped so a transparent fill costs nothing.
    pub fn new_filled(width: u32, height: u32, color: Rgba<u8>) -> Self {
        let mut img = Self::new(width, height);
        if color[3] > 0 {
            img.fill(color);
        }
        img
    }

    /// Import from a flat `RgbaImage`.  Only non-transparent chunks are stored.
    /// Chunk conversion is parallelised with rayon for faster import of large images.
    pub fn from_rgba_image(src: &RgbaImage) -> Self {
        let width = src.width();
        let height = src.height();
        let mut img = Self::new(width, height);

        let chunks_x = img.chunks_per_row as usize;
        let chunks_y = height.div_ceil(CHUNK_SIZE) as usize;
        let total_chunks = chunks_x * chunks_y;
        let src_raw = src.as_raw();

        let chunk_results: Vec<(usize, Option<Arc<RgbaImage>>)> = (0..total_chunks)
            .into_par_iter()
            .map(|flat| {
                let cx = (flat % chunks_x) as u32;
                let cy = (flat / chunks_x) as u32;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                let cw = CHUNK_SIZE.min(width - base_x);
                let ch = CHUNK_SIZE.min(height - base_y);
                let chunk_stride = CHUNK_SIZE as usize * 4;
                let mut chunk_data = vec![0u8; chunk_stride * CHUNK_SIZE as usize];
                let mut has_content = false;

                for ly in 0..ch {
                    let src_start = ((base_y + ly) * width + base_x) as usize * 4;
                    let dst_start = ly as usize * chunk_stride;
                    let byte_len = cw as usize * 4;
                    chunk_data[dst_start..dst_start + byte_len]
                        .copy_from_slice(&src_raw[src_start..src_start + byte_len]);

                    if !has_content {
                        for lx in 0..cw as usize {
                            if chunk_data[dst_start + lx * 4 + 3] != 0 {
                                has_content = true;
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    let chunk = RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk_data).unwrap();
                    (flat, Some(Arc::new(chunk)))
                } else {
                    (flat, None)
                }
            })
            .collect();

        for (idx, chunk) in chunk_results {
            img.chunks[idx] = chunk;
        }
        img
    }

    /// Import from a flat RGBA byte slice without creating an intermediate `RgbaImage`.
    /// `data` must be exactly `width * height * 4` bytes (row-major, RGBA).
    /// Only non-transparent chunks are stored.  Parallelised with rayon.
    pub fn from_raw_rgba(width: u32, height: u32, data: &[u8]) -> Self {
        debug_assert_eq!(data.len(), (width as usize) * (height as usize) * 4);
        let mut img = Self::new(width, height);

        let chunks_x = img.chunks_per_row as usize;
        let chunks_y = height.div_ceil(CHUNK_SIZE) as usize;
        let total_chunks = chunks_x * chunks_y;

        let chunk_results: Vec<(usize, Option<Arc<RgbaImage>>)> = (0..total_chunks)
            .into_par_iter()
            .map(|flat| {
                let cx = (flat % chunks_x) as u32;
                let cy = (flat / chunks_x) as u32;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                let cw = CHUNK_SIZE.min(width - base_x);
                let ch = CHUNK_SIZE.min(height - base_y);
                let chunk_stride = CHUNK_SIZE as usize * 4;
                let mut chunk_data = vec![0u8; chunk_stride * CHUNK_SIZE as usize];
                let mut has_content = false;

                for ly in 0..ch {
                    let src_start = ((base_y + ly) * width + base_x) as usize * 4;
                    let dst_start = ly as usize * chunk_stride;
                    let byte_len = cw as usize * 4;
                    chunk_data[dst_start..dst_start + byte_len]
                        .copy_from_slice(&data[src_start..src_start + byte_len]);

                    if !has_content {
                        for lx in 0..cw as usize {
                            if chunk_data[dst_start + lx * 4 + 3] != 0 {
                                has_content = true;
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    let chunk = RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk_data).unwrap();
                    (flat, Some(Arc::new(chunk)))
                } else {
                    (flat, None)
                }
            })
            .collect();

        for (idx, chunk) in chunk_results {
            img.chunks[idx] = chunk;
        }
        img
    }

    /// Build a TiledImage from a sub-region RGBA buffer.
    ///
    /// `data` is `region_w * region_h * 4` bytes, positioned at `(off_x, off_y)`
    /// within a `canvas_w × canvas_h` canvas. Only chunks overlapping the region
    /// are created (parallelized with rayon). Much faster than per-pixel `put_pixel`.
    pub fn from_region_rgba(
        canvas_w: u32,
        canvas_h: u32,
        data: &[u8],
        region_w: u32,
        region_h: u32,
        off_x: i32,
        off_y: i32,
    ) -> Self {
        let mut img = Self::new(canvas_w, canvas_h);
        if region_w == 0 || region_h == 0 || data.is_empty() {
            return img;
        }

        let chunks_x = img.chunks_per_row as usize;

        // Compute chunk coordinate range that overlaps the region
        let rx0 = off_x.max(0) as u32;
        let ry0 = off_y.max(0) as u32;
        let rx1 = ((off_x + region_w as i32) as u32).min(canvas_w);
        let ry1 = ((off_y + region_h as i32) as u32).min(canvas_h);

        let cx_start = (rx0 / CHUNK_SIZE) as usize;
        let cx_end = rx1.div_ceil(CHUNK_SIZE) as usize;
        let cy_start = (ry0 / CHUNK_SIZE) as usize;
        let cy_end = ry1.div_ceil(CHUNK_SIZE) as usize;

        // Collect all chunk indices that overlap the region
        let mut overlapping: Vec<usize> =
            Vec::with_capacity((cx_end - cx_start) * (cy_end - cy_start));
        for cy in cy_start..cy_end {
            for cx in cx_start..cx_end {
                overlapping.push(cy * chunks_x + cx);
            }
        }

        let region_stride = region_w as usize * 4;
        let chunk_results: Vec<(usize, Option<Arc<RgbaImage>>)> = overlapping
            .into_par_iter()
            .map(|flat| {
                let cx = (flat % chunks_x) as u32;
                let cy = (flat / chunks_x) as u32;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                let cw = CHUNK_SIZE.min(canvas_w - base_x);
                let ch = CHUNK_SIZE.min(canvas_h - base_y);
                let chunk_stride = CHUNK_SIZE as usize * 4;
                let mut chunk_data = vec![0u8; chunk_stride * CHUNK_SIZE as usize];
                let mut has_content = false;

                for ly in 0..ch {
                    let gy = base_y + ly;
                    let ry = gy as i32 - off_y;
                    if ry < 0 || ry >= region_h as i32 {
                        continue;
                    }

                    // Compute horizontal overlap
                    let gx_start = base_x;
                    let gx_end = base_x + cw;
                    let rx_start = (gx_start as i32 - off_x).max(0) as u32;
                    let rx_end = ((gx_end as i32 - off_x) as u32).min(region_w);
                    if rx_start >= rx_end {
                        continue;
                    }

                    let lx_start = (off_x + rx_start as i32) as u32 - base_x;
                    let copy_w = rx_end - rx_start;

                    let src_start = ry as usize * region_stride + rx_start as usize * 4;
                    let dst_start = ly as usize * chunk_stride + lx_start as usize * 4;
                    let byte_len = copy_w as usize * 4;

                    chunk_data[dst_start..dst_start + byte_len]
                        .copy_from_slice(&data[src_start..src_start + byte_len]);

                    if !has_content {
                        for px in 0..copy_w as usize {
                            if chunk_data[dst_start + px * 4 + 3] != 0 {
                                has_content = true;
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    let chunk = RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk_data).unwrap();
                    (flat, Some(Arc::new(chunk)))
                } else {
                    (flat, None)
                }
            })
            .collect();

        for (idx, chunk) in chunk_results {
            img.chunks[idx] = chunk;
        }
        img
    }

    /// Flatten back to a contiguous `RgbaImage`.
    pub fn to_rgba_image(&self) -> RgbaImage {
        let mut out = RgbaImage::new(self.width, self.height);
        let out_raw = out.as_mut();
        let out_stride = self.width as usize * 4;
        for (cx, cy) in self.chunk_keys() {
            if let Some(chunk) = self.get_chunk(cx, cy) {
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;
                let cw = (CHUNK_SIZE.min(self.width.saturating_sub(base_x))) as usize;
                let ch = CHUNK_SIZE.min(self.height.saturating_sub(base_y));
                let chunk_raw = chunk.as_raw();
                let chunk_stride = CHUNK_SIZE as usize * 4;
                for ly in 0..ch as usize {
                    let src_start = ly * chunk_stride;
                    let src_end = src_start + cw * 4;
                    let dst_start = (base_y as usize + ly) * out_stride + base_x as usize * 4;
                    let dst_end = dst_start + cw * 4;
                    out_raw[dst_start..dst_end].copy_from_slice(&chunk_raw[src_start..src_end]);
                }
            }
        }
        out
    }

    // ---- chunk-level flip / rotate (avoid full-image materialisation) --------

    /// Flip horizontally without materialising the full image.
    /// Iterates source chunks and writes transformed pixels to a new chunk array.
    pub fn flip_horizontal_chunked(&mut self) {
        let cs = CHUNK_SIZE;
        let w = self.width;
        let cpr = self.chunks_per_row;
        let total = self.chunks.len();
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % cpr;
                let src_cy = (src_idx as u32) / cpr;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(w - base_x);
                let ch = cs.min(self.height - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 {
                            continue;
                        }

                        let dst_x = w - 1 - (base_x + lx);
                        let dst_cx = dst_x / cs;
                        let dst_lx = dst_x % cs;
                        let dst_i = (src_cy * cpr + dst_cx) as usize;

                        let dc = Arc::make_mut(
                            dst[dst_i].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))),
                        );
                        dc.put_pixel(
                            dst_lx,
                            ly,
                            Rgba([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]),
                        );
                    }
                }
            }
        }
        self.chunks = dst;
    }

    /// Flip vertically without materialising the full image.
    pub fn flip_vertical_chunked(&mut self) {
        let cs = CHUNK_SIZE;
        let h = self.height;
        let cpr = self.chunks_per_row;
        let total = self.chunks.len();
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % cpr;
                let src_cy = (src_idx as u32) / cpr;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(self.width - base_x);
                let ch = cs.min(h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    let dst_y = h - 1 - (base_y + ly);
                    let dst_cy = dst_y / cs;
                    let dst_ly = dst_y % cs;
                    let dst_i = (dst_cy * cpr + src_cx) as usize;

                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 {
                            continue;
                        }

                        let dc = Arc::make_mut(
                            dst[dst_i].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))),
                        );
                        dc.put_pixel(
                            lx,
                            dst_ly,
                            Rgba([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]),
                        );
                    }
                }
            }
        }
        self.chunks = dst;
    }

    /// Rotate 180° without materialising the full image (= H flip + V flip combined).
    pub fn rotate_180_chunked(&mut self) {
        let cs = CHUNK_SIZE;
        let w = self.width;
        let h = self.height;
        let cpr = self.chunks_per_row;
        let total = self.chunks.len();
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % cpr;
                let src_cy = (src_idx as u32) / cpr;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(w - base_x);
                let ch = cs.min(h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    let dst_y = h - 1 - (base_y + ly);
                    let dst_cy = dst_y / cs;
                    let dst_ly = dst_y % cs;

                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 {
                            continue;
                        }

                        let dst_x = w - 1 - (base_x + lx);
                        let dst_cx = dst_x / cs;
                        let dst_lx = dst_x % cs;
                        let dst_i = (dst_cy * cpr + dst_cx) as usize;

                        let dc = Arc::make_mut(
                            dst[dst_i].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))),
                        );
                        dc.put_pixel(
                            dst_lx,
                            dst_ly,
                            Rgba([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]),
                        );
                    }
                }
            }
        }
        self.chunks = dst;
    }

    /// Rotate 90° CW without materialising the full image.
    /// Returns a new TiledImage with swapped dimensions (W×H → H×W).
    pub fn rotate_90cw_chunked(&self) -> TiledImage {
        let cs = CHUNK_SIZE;
        let old_w = self.width;
        let old_h = self.height;
        let new_w = old_h;
        let new_h = old_w;
        let new_cpr = new_w.div_ceil(cs);
        let new_cpc = new_h.div_ceil(cs);
        let total = (new_cpr * new_cpc) as usize;
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % self.chunks_per_row;
                let src_cy = (src_idx as u32) / self.chunks_per_row;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(old_w - base_x);
                let ch = cs.min(old_h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 {
                            continue;
                        }
                        // 90° CW: (x, y) → (old_h - 1 - y, x)
                        let dx = old_h - 1 - (base_y + ly);
                        let dy = base_x + lx;
                        let dcx = dx / cs;
                        let dcy = dy / cs;
                        let dlx = dx % cs;
                        let dly = dy % cs;
                        let di = (dcy * new_cpr + dcx) as usize;

                        let dc = Arc::make_mut(
                            dst[di].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))),
                        );
                        dc.put_pixel(
                            dlx,
                            dly,
                            Rgba([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]),
                        );
                    }
                }
            }
        }

        TiledImage {
            width: new_w,
            height: new_h,
            chunks_per_row: new_cpr,
            chunks: dst,
        }
    }

    /// Rotate 90° CCW without materialising the full image.
    /// Returns a new TiledImage with swapped dimensions (W×H → H×W).
    pub fn rotate_90ccw_chunked(&self) -> TiledImage {
        let cs = CHUNK_SIZE;
        let old_w = self.width;
        let old_h = self.height;
        let new_w = old_h;
        let new_h = old_w;
        let new_cpr = new_w.div_ceil(cs);
        let new_cpc = new_h.div_ceil(cs);
        let total = (new_cpr * new_cpc) as usize;
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % self.chunks_per_row;
                let src_cy = (src_idx as u32) / self.chunks_per_row;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(old_w - base_x);
                let ch = cs.min(old_h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 {
                            continue;
                        }
                        // 90° CCW: (x, y) → (y, old_w - 1 - x)
                        let dx = base_y + ly;
                        let dy = old_w - 1 - (base_x + lx);
                        let dcx = dx / cs;
                        let dcy = dy / cs;
                        let dlx = dx % cs;
                        let dly = dy % cs;
                        let di = (dcy * new_cpr + dcx) as usize;

                        let dc = Arc::make_mut(
                            dst[di].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))),
                        );
                        dc.put_pixel(
                            dlx,
                            dly,
                            Rgba([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]),
                        );
                    }
                }
            }
        }

        TiledImage {
            width: new_w,
            height: new_h,
            chunks_per_row: new_cpr,
            chunks: dst,
        }
    }

    /// Extract a sub-rectangle of the image as raw RGBA bytes (tightly packed,
    /// `rect_w * rect_h * 4` bytes).  Used for partial GPU texture uploads.
    pub fn extract_region_rgba(&self, rx: u32, ry: u32, rw: u32, rh: u32) -> Vec<u8> {
        let size = (rw as u64) * (rh as u64) * 4;
        if size > 1_073_741_824 {
            return Vec::new();
        } // 1GB sanity limit
        let mut buf = vec![0u8; size as usize];
        for y in 0..rh {
            let iy = ry + y;
            if iy >= self.height {
                continue;
            }
            for x in 0..rw {
                let ix = rx + x;
                if ix >= self.width {
                    continue;
                }
                let px = self.get_pixel(ix, iy);
                let off = ((y * rw + x) * 4) as usize;
                buf[off] = px[0];
                buf[off + 1] = px[1];
                buf[off + 2] = px[2];
                buf[off + 3] = px[3];
            }
        }
        buf
    }

    /// Fast chunk-aware region extraction.  Instead of per-pixel `get_pixel()`
    /// calls, iterates only overlapping chunks and copies rows via `memcpy`.
    /// For a 500×500 stroke region on a 5000×5000 canvas this is ~100× faster
    /// than `extract_region_rgba` / `to_rgba_image`.
    pub fn extract_region_rgba_fast(&self, rx: u32, ry: u32, rw: u32, rh: u32, buf: &mut Vec<u8>) {
        let size = (rw as u64) * (rh as u64) * 4;
        if size > 1_073_741_824 {
            buf.clear();
            return;
        } // 1GB sanity limit
        let needed = size as usize;
        buf.resize(needed, 0);
        // Zero the buffer (un-populated chunks must be transparent)
        // Using a fast fill instead of allocating a new Vec each frame.
        for b in buf.iter_mut() {
            *b = 0;
        }

        let cx_start = rx / CHUNK_SIZE;
        let cx_end = (rx + rw).div_ceil(CHUNK_SIZE);
        let cy_start = ry / CHUNK_SIZE;
        let cy_end = (ry + rh).div_ceil(CHUNK_SIZE);

        for cy in cy_start..cy_end {
            for cx in cx_start..cx_end {
                let chunk = match self.get_chunk(cx, cy) {
                    Some(c) => c,
                    None => continue, // transparent – already zeroed
                };
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                // Overlap between this chunk and the requested region
                let ox = rx.max(base_x);
                let oy = ry.max(base_y);
                let ox2 = (rx + rw).min(base_x + CHUNK_SIZE).min(self.width);
                let oy2 = (ry + rh).min(base_y + CHUNK_SIZE).min(self.height);
                if ox >= ox2 || oy >= oy2 {
                    continue;
                }

                let ow = (ox2 - ox) as usize;
                let chunk_raw = chunk.as_raw();
                let chunk_stride = CHUNK_SIZE as usize * 4;

                for sy in oy..oy2 {
                    let lx = (ox - base_x) as usize;
                    let ly = (sy - base_y) as usize;
                    let src_start = ly * chunk_stride + lx * 4;
                    let src_end = src_start + ow * 4;

                    let dx = (ox - rx) as usize;
                    let dy = (sy - ry) as usize;
                    let dst_start = dy * (rw as usize) * 4 + dx * 4;
                    let dst_end = dst_start + ow * 4;

                    buf[dst_start..dst_end].copy_from_slice(&chunk_raw[src_start..src_end]);
                }
            }
        }
    }

    // ---- indexing helpers ----------------------------------------------------

    #[inline(always)]
    fn flat_index(&self, cx: u32, cy: u32) -> usize {
        (cy * self.chunks_per_row + cx) as usize
    }

    #[inline(always)]
    fn chunk_coord(x: u32, y: u32) -> (u32, u32) {
        (x / CHUNK_SIZE, y / CHUNK_SIZE)
    }

    #[inline(always)]
    fn local(x: u32, y: u32) -> (u32, u32) {
        (x % CHUNK_SIZE, y % CHUNK_SIZE)
    }

    // ---- pixel access -------------------------------------------------------

    /// Read a pixel (returns `&TRANSPARENT_PIXEL` for missing chunks).
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> &Rgba<u8> {
        if x >= self.width || y >= self.height {
            return &TRANSPARENT_PIXEL;
        }
        let (cx, cy) = Self::chunk_coord(x, y);
        let (lx, ly) = Self::local(x, y);
        let idx = self.flat_index(cx, cy);
        self.chunks[idx]
            .as_ref()
            .map(|c| c.get_pixel(lx, ly))
            .unwrap_or(&TRANSPARENT_PIXEL)
    }

    /// Write a pixel (creates the chunk on demand, COW-clones if shared).
    #[inline]
    pub fn put_pixel(&mut self, x: u32, y: u32, pixel: Rgba<u8>) {
        if x >= self.width || y >= self.height {
            return;
        }
        let (cx, cy) = Self::chunk_coord(x, y);
        let (lx, ly) = Self::local(x, y);
        let idx = self.flat_index(cx, cy);
        let arc = self.chunks[idx]
            .get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
        Arc::make_mut(arc).put_pixel(lx, ly, pixel);
    }

    /// Blit an RGBA sub-image at a given position using bulk chunk row copies.
    /// Much faster than per-pixel put_pixel for large regions.
    pub fn blit_rgba_at(&mut self, dst_x: i32, dst_y: i32, src_w: u32, src_h: u32, data: &[u8]) {
        debug_assert_eq!(data.len(), src_w as usize * src_h as usize * 4);
        let cs = CHUNK_SIZE;

        for sy in 0..src_h {
            let gy = dst_y + sy as i32;
            if gy < 0 || gy as u32 >= self.height {
                continue;
            }
            let gy = gy as u32;

            let src_row_start = sy as usize * src_w as usize * 4;

            // Process contiguous runs of pixels in this row
            let mut sx = 0u32;
            while sx < src_w {
                let gx = dst_x + sx as i32;
                if gx < 0 {
                    sx += 1;
                    continue;
                }
                let gx = gx as u32;
                if gx >= self.width {
                    break;
                }

                let (cx, cy) = Self::chunk_coord(gx, gy);
                let (lx, ly) = Self::local(gx, gy);
                let idx = self.flat_index(cx, cy);

                // How many pixels can we write into this chunk row?
                let chunk_remaining = cs - lx;
                let src_remaining = src_w - sx;
                let canvas_remaining = self.width - gx;
                let run = chunk_remaining.min(src_remaining).min(canvas_remaining);

                // Check if this run has any non-transparent pixels
                let src_off = src_row_start + sx as usize * 4;
                let byte_len = run as usize * 4;
                let has_content = data[src_off..src_off + byte_len]
                    .chunks_exact(4)
                    .any(|px| px[3] != 0);

                if has_content {
                    let arc =
                        self.chunks[idx].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs)));
                    let chunk = Arc::make_mut(arc);
                    let dst_off = (ly as usize * cs as usize + lx as usize) * 4;
                    chunk.as_mut()[dst_off..dst_off + byte_len]
                        .copy_from_slice(&data[src_off..src_off + byte_len]);
                }

                sx += run;
            }
        }
    }

    /// Blit RGBA data with full replacement including transparent pixels.
    /// Unlike `blit_rgba_at`, this writes ALL pixels in the source region,
    /// including fully-transparent ones. Used by fill preview GPU path where
    /// the dirty region must be fully overwritten (unfilled pixels → transparent).
    pub fn blit_rgba_at_replace(
        &mut self,
        dst_x: i32,
        dst_y: i32,
        src_w: u32,
        src_h: u32,
        data: &[u8],
    ) {
        debug_assert_eq!(data.len(), src_w as usize * src_h as usize * 4);
        let cs = CHUNK_SIZE;

        for sy in 0..src_h {
            let gy = dst_y + sy as i32;
            if gy < 0 || gy as u32 >= self.height {
                continue;
            }
            let gy = gy as u32;

            let src_row_start = sy as usize * src_w as usize * 4;

            let mut sx = 0u32;
            while sx < src_w {
                let gx = dst_x + sx as i32;
                if gx < 0 {
                    sx += 1;
                    continue;
                }
                let gx = gx as u32;
                if gx >= self.width {
                    break;
                }

                let (_cx, _cy) = Self::chunk_coord(gx, gy);
                let (lx, ly) = Self::local(gx, gy);
                let idx = self.flat_index(_cx, _cy);

                let chunk_remaining = cs - lx;
                let src_remaining = src_w - sx;
                let canvas_remaining = self.width - gx;
                let run = chunk_remaining.min(src_remaining).min(canvas_remaining);

                let src_off = src_row_start + sx as usize * 4;
                let byte_len = run as usize * 4;

                // Check if entire run is transparent
                let all_transparent = data[src_off..src_off + byte_len]
                    .chunks_exact(4)
                    .all(|px| px[3] == 0);

                if all_transparent {
                    // If chunk exists, clear these pixels to transparent
                    if let Some(arc) = &mut self.chunks[idx] {
                        let chunk = Arc::make_mut(arc);
                        let dst_off = (ly as usize * cs as usize + lx as usize) * 4;
                        for b in &mut chunk.as_mut()[dst_off..dst_off + byte_len] {
                            *b = 0;
                        }
                    }
                    // If chunk doesn't exist, already transparent — nothing to do
                } else {
                    let arc =
                        self.chunks[idx].get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs)));
                    let chunk = Arc::make_mut(arc);
                    let dst_off = (ly as usize * cs as usize + lx as usize) * 4;
                    chunk.as_mut()[dst_off..dst_off + byte_len]
                        .copy_from_slice(&data[src_off..src_off + byte_len]);
                }

                sx += run;
            }
        }
    }

    /// Mutable reference to a pixel (creates the chunk on demand, COW-clones if shared).
    #[inline]
    pub fn get_pixel_mut(&mut self, x: u32, y: u32) -> &mut Rgba<u8> {
        let (cx, cy) = Self::chunk_coord(x, y);
        let (lx, ly) = Self::local(x, y);
        let idx = self.flat_index(cx, cy);
        let arc = self.chunks[idx]
            .get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
        Arc::make_mut(arc).get_pixel_mut(lx, ly)
    }

    /// Read-only access to a chunk (if it exists).
    pub fn get_chunk(&self, cx: u32, cy: u32) -> Option<&RgbaImage> {
        let idx = self.flat_index(cx, cy);
        self.chunks.get(idx).and_then(|c| c.as_deref())
    }

    /// Mutable access to an existing chunk (COW-clones if shared).
    pub fn get_chunk_mut(&mut self, cx: u32, cy: u32) -> Option<&mut RgbaImage> {
        let idx = self.flat_index(cx, cy);
        self.chunks
            .get_mut(idx)
            .and_then(|slot| slot.as_mut())
            .map(Arc::make_mut)
    }

    /// Get or create a chunk, returning a mutable reference (COW-safe).
    pub fn ensure_chunk_mut(&mut self, cx: u32, cy: u32) -> &mut RgbaImage {
        let idx = self.flat_index(cx, cy);
        let arc = self.chunks[idx]
            .get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
        Arc::make_mut(arc)
    }

    /// Place a fully-built chunk at the given chunk coordinate.
    pub fn set_chunk(&mut self, cx: u32, cy: u32, chunk: RgbaImage) {
        let idx = self.flat_index(cx, cy);
        if idx < self.chunks.len() {
            self.chunks[idx] = Some(Arc::new(chunk));
        }
    }

    /// Iterator over populated chunk coordinates.
    pub fn chunk_keys(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let cpr = self.chunks_per_row;
        self.chunks.iter().enumerate().filter_map(move |(i, slot)| {
            if slot.is_some() {
                Some(((i as u32) % cpr, (i as u32) / cpr))
            } else {
                None
            }
        })
    }

    /// Number of populated chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_some()).count()
    }

    // ---- bulk operations ----------------------------------------------------

    /// Transform populated chunks in parallel without flattening the image.
    /// The callback must preserve zero alpha so sparse transparent chunks can
    /// remain unallocated.
    pub fn par_map_populated<F>(&mut self, transform: F)
    where
        F: Fn(u32, u32, Rgba<u8>) -> Rgba<u8> + Sync,
    {
        let chunks_per_row = self.chunks_per_row;
        let width = self.width;
        let height = self.height;
        self.chunks
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, slot)| {
                let Some(arc) = slot.as_mut() else {
                    return;
                };
                let cx = index as u32 % chunks_per_row;
                let cy = index as u32 / chunks_per_row;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;
                let valid_w = CHUNK_SIZE.min(width - base_x);
                let valid_h = CHUNK_SIZE.min(height - base_y);
                let chunk = Arc::make_mut(arc);
                for ly in 0..valid_h {
                    for lx in 0..valid_w {
                        let pixel = *chunk.get_pixel(lx, ly);
                        chunk.put_pixel(lx, ly, transform(base_x + lx, base_y + ly, pixel));
                    }
                }
            });
    }

    /// Fill every pixel with `color`.
    pub fn fill(&mut self, color: Rgba<u8>) {
        for slot in &mut self.chunks {
            let arc = slot.get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
            let chunk = Arc::make_mut(arc);
            for pixel in chunk.pixels_mut() {
                *pixel = color;
            }
        }
    }

    /// Drop all chunks (make the image fully transparent).
    pub fn clear(&mut self) {
        for slot in &mut self.chunks {
            *slot = None;
        }
    }

    /// Drop chunks whose coordinates overlap the given pixel-space rectangle.
    pub fn clear_region(&mut self, min_x: u32, min_y: u32, max_x: u32, max_y: u32) {
        let min_cx = min_x / CHUNK_SIZE;
        let max_cx = max_x.div_ceil(CHUNK_SIZE);
        let min_cy = min_y / CHUNK_SIZE;
        let max_cy = max_y.div_ceil(CHUNK_SIZE);
        let total_cx = self.chunks_per_row;
        let total_cy = self.height.div_ceil(CHUNK_SIZE);
        for cy in min_cy..max_cy.min(total_cy) {
            for cx in min_cx..max_cx.min(total_cx) {
                let idx = (cy * total_cx + cx) as usize;
                if idx < self.chunks.len() {
                    self.chunks[idx] = None;
                }
            }
        }
    }

    /// Width accessor (matches `RgbaImage::width()`).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height accessor (matches `RgbaImage::height()`).
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Approximate memory usage in bytes.
    /// Shared (COW) chunks are counted at minimal cost (Arc pointer only)
    /// since their pixel data is shared with undo snapshots.
    pub fn memory_bytes(&self) -> usize {
        let chunk_byte_size = (CHUNK_SIZE * CHUNK_SIZE * 4) as usize;
        self.chunks
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|arc| {
                if Arc::strong_count(arc) == 1 {
                    chunk_byte_size
                } else {
                    // Shared with snapshots — only count the Arc pointer overhead
                    std::mem::size_of::<usize>() * 2
                }
            })
            .sum()
    }

    /// Total pixel memory owned by this image (ignoring sharing).
    /// Used for diagnostic display.
    pub fn memory_bytes_total(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_some()).count() * (CHUNK_SIZE * CHUNK_SIZE * 4) as usize
    }
}
