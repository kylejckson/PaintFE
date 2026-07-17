// ============================================================================
// TEXTURE POOL — recycle GPU textures to avoid create/destroy churn
// ============================================================================

use std::collections::HashMap;

/// Key for pooled textures: (width, height, mip_levels).
type PoolKey = (u32, u32, u32);

/// A pool of GPU textures keyed by dimensions.
///
/// When a layer is deleted, its texture goes back into the pool.  When a new
/// layer of the same size is created, we grab a texture from the pool instead
/// of allocating a new one.
///
/// This avoids the overhead of `device.create_texture` every frame and reduces
/// driver-side memory fragmentation.
pub struct TexturePool {
    pool: HashMap<PoolKey, Vec<wgpu::Texture>>,
    /// Maximum number of textures to keep per key.
    max_per_key: usize,
    /// Global cap prevents several 4K/8K size classes retaining huge VRAM.
    max_bytes: usize,
}

impl TexturePool {
    pub fn new() -> Self {
        Self {
            pool: HashMap::new(),
            max_per_key: 2,
            max_bytes: 128 * 1024 * 1024,
        }
    }

    /// Return a recycled texture if one exists for the given dimensions,
    /// otherwise return `None` and the caller should create a new one.
    pub fn acquire(&mut self, width: u32, height: u32, mip_levels: u32) -> Option<wgpu::Texture> {
        let key: PoolKey = (width, height, mip_levels);
        self.pool.get_mut(&key).and_then(|v| v.pop())
    }

    /// Return a texture to the pool for future reuse.
    /// If the pool is full for this key, the texture is simply dropped.
    pub fn release(&mut self, texture: wgpu::Texture, width: u32, height: u32, mip_levels: u32) {
        let key: PoolKey = (width, height, mip_levels);
        let texture_bytes = texture_bytes(width, height, mip_levels);
        if self.pooled_memory_bytes().saturating_add(texture_bytes) > self.max_bytes {
            return;
        }
        let entry = self.pool.entry(key).or_default();
        if entry.len() < self.max_per_key {
            entry.push(texture);
        }
        // else: texture is dropped here, freeing GPU memory
    }

    /// Drop all pooled textures (e.g., on context loss or shutdown).
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Total number of textures currently in the pool.
    pub fn pooled_count(&self) -> usize {
        self.pool.values().map(|v| v.len()).sum()
    }

    /// Approximate GPU memory held by pooled textures (bytes).
    pub fn pooled_memory_bytes(&self) -> usize {
        self.pool
            .iter()
            .map(|((w, h, mip_levels), textures)| {
                let bytes_per = texture_bytes(*w, *h, *mip_levels);
                bytes_per * textures.len()
            })
            .sum()
    }
}

fn texture_bytes(mut width: u32, mut height: u32, mip_levels: u32) -> usize {
    let mut pixels = 0usize;
    for _ in 0..mip_levels {
        pixels = pixels.saturating_add(width as usize * height as usize);
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    pixels.saturating_mul(4)
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new()
    }
}
