/// Warp through physical reference cells, including nonuniform reference meshes.
/// Each row owns its output bytes; source samples use premultiplied interpolation.
pub fn warp_reference_mesh(
    source: &RgbaImage,
    original: &[[f32; 2]],
    deformed: &[[f32; 2]],
    cols: usize,
    rows: usize,
) -> RgbaImage {
    if source.width() == 0
        || source.height() == 0
        || original == deformed
        || cols == 0
        || rows == 0
        || cols > 32
        || rows > 32
        || original.len() != (cols + 1) * (rows + 1)
        || original.len() != deformed.len()
        || original
            .iter()
            .chain(deformed)
            .flatten()
            .any(|v| !v.is_finite())
    {
        return source.clone();
    }
    type Triangle = ([[f32; 2]; 3], [[f32; 2]; 3]);
    let subdivisions = 4;
    let fine_cols = cols * subdivisions;
    let fine_rows = rows * subdivisions;
    let mut triangles: Vec<Triangle> = Vec::with_capacity(fine_cols * fine_rows * 2);
    for row in 0..fine_rows {
        for col in 0..fine_cols {
            let uv = [
                [col as f32, row as f32],
                [col as f32 + 1., row as f32],
                [col as f32 + 1., row as f32 + 1.],
                [col as f32, row as f32 + 1.],
            ];
            let map = |points: &[[f32; 2]]| {
                uv.map(|[u, v]| {
                    catmull_rom_surface(
                        points,
                        cols,
                        rows,
                        u / subdivisions as f32,
                        v / subdivisions as f32,
                    )
                })
            };
            let a = map(original);
            let b = map(deformed);
            for indices in [[0, 1, 2], [0, 2, 3]] {
                triangles.push((indices.map(|i| a[i]), indices.map(|i| b[i])));
            }
        }
    }
    let (width, height) = source.dimensions();
    let mut output = source.clone();
    for clear in [true, false] {
        let mut bins = vec![Vec::new(); height as usize];
        for (index, (a, b)) in triangles.iter().enumerate() {
            let t = if clear { a } else { b };
            let min = t
                .iter()
                .map(|p| p[1])
                .fold(f32::INFINITY, f32::min)
                .floor()
                .max(0.) as u32;
            let max = t
                .iter()
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .max(0.) as u32;
            for y in min.min(height)..max.min(height) {
                bins[y as usize].push(index);
            }
        }
        output
            .as_mut()
            .par_chunks_mut(width as usize * 4)
            .enumerate()
            .for_each(|(y, row)| {
                let py = y as f32 + 0.5;
                for &index in &bins[y] {
                    let (a, b) = triangles[index];
                    let t = if clear { a } else { b };
                    let [p, q, r] = t;
                    let det = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
                    if det.abs() < 1e-7 {
                        continue;
                    }
                    let min = t
                        .iter()
                        .map(|p| p[0])
                        .fold(f32::INFINITY, f32::min)
                        .floor()
                        .max(0.) as u32;
                    let max = t
                        .iter()
                        .map(|p| p[0])
                        .fold(f32::NEG_INFINITY, f32::max)
                        .ceil()
                        .max(0.) as u32;
                    for x in min.min(width)..max.min(width) {
                        let px = x as f32 + 0.5;
                        let v = ((px - p[0]) * (r[1] - p[1]) - (py - p[1]) * (r[0] - p[0])) / det;
                        let w = ((q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])) / det;
                        let u = 1. - v - w;
                        if u < -1e-5 || v < -1e-5 || w < -1e-5 {
                            continue;
                        }
                        let offset = x as usize * 4;
                        let pixel = if clear {
                            [0; 4]
                        } else {
                            mesh_sample(
                                source,
                                u * a[0][0] + v * a[1][0] + w * a[2][0] - 0.5,
                                u * a[0][1] + v * a[1][1] + w * a[2][1] - 0.5,
                            )
                        };
                        row[offset..offset + 4].copy_from_slice(&pixel);
                    }
                }
            });
    }
    output
}

fn mesh_sample(image: &RgbaImage, x: f32, y: f32) -> [u8; 4] {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let mut sum = [0.; 4];
    for (dx, dy, weight) in [
        (0, 0, (1. - fx) * (1. - fy)),
        (1, 0, fx * (1. - fy)),
        (0, 1, (1. - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let sx = ix + dx;
        let sy = iy + dy;
        if sx < 0 || sy < 0 || sx >= image.width() as i64 || sy >= image.height() as i64 {
            continue;
        }
        let p = image.get_pixel(sx as u32, sy as u32).0;
        let alpha = p[3] as f32 / 255.;
        for c in 0..3 {
            sum[c] += p[c] as f32 * alpha * weight;
        }
        sum[3] += alpha * weight;
    }
    if sum[3] <= 1e-7 {
        return [0; 4];
    }
    [
        (sum[0] / sum[3]).round().clamp(0., 255.) as u8,
        (sum[1] / sum[3]).round().clamp(0., 255.) as u8,
        (sum[2] / sum[3]).round().clamp(0., 255.) as u8,
        (sum[3] * 255.).round().clamp(0., 255.) as u8,
    ]
}

impl DisplacementField {
    /// Restore the reference image locally without resampling intermediate images.
    pub fn reconstruct(&mut self, cx: f32, cy: f32, radius: f32, strength: f32) {
        let radius = radius.max(1.);
        let x0 = (cx - radius).floor().max(0.) as u32;
        let y0 = (cy - radius).floor().max(0.) as u32;
        let x1 = ((cx + radius).ceil().max(0.) as u32).min(self.width);
        let y1 = ((cy + radius).ceil().max(0.) as u32).min(self.height);
        for y in y0..y1 {
            for x in x0..x1 {
                let distance = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / radius;
                if distance >= 1. {
                    continue;
                }
                let keep = 1. - strength.clamp(0., 1.) * (1. - distance).powi(2);
                let i = (y as usize * self.width as usize + x as usize) * 2;
                self.data[i] *= keep;
                self.data[i + 1] *= keep;
            }
        }
    }
}
