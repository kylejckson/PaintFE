const FIELD_TILE: u32 = 32;
const FIELD_HISTORY_LIMIT: usize = 128 * 1024 * 1024;

#[derive(Clone, Debug)]
struct FieldTileChange {
    x: u32,
    y: u32,
    before: Vec<f32>,
    after: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct LiquifyAdjustmentHistory {
    pending: std::collections::BTreeMap<(u32, u32), Vec<f32>>,
    undo: Vec<Vec<FieldTileChange>>,
    redo: Vec<Vec<FieldTileChange>>,
    pub limited: bool,
    pending_bytes: usize,
    stored_bytes: usize,
    limit: usize,
}

impl Default for LiquifyAdjustmentHistory {
    fn default() -> Self {
        Self {
            pending: Default::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            limited: false,
            pending_bytes: 0,
            stored_bytes: 0,
            limit: FIELD_HISTORY_LIMIT,
        }
    }
}
fn field_changes_bytes(changes: &[FieldTileChange]) -> usize {
    changes
        .iter()
        .map(|tile| {
            (tile.before.len() + tile.after.len()) * 4 + std::mem::size_of::<FieldTileChange>()
        })
        .sum()
}

fn field_tile(field: &crate::ops::transform::DisplacementField, x: u32, y: u32) -> Vec<f32> {
    let mut result = Vec::with_capacity(
        ((x + FIELD_TILE).min(field.width) - x) as usize
            * ((y + FIELD_TILE).min(field.height) - y) as usize
            * 2,
    );
    for row in y..(y + FIELD_TILE).min(field.height) {
        let start = (row as usize * field.width as usize + x as usize) * 2;
        let count = ((x + FIELD_TILE).min(field.width) - x) as usize * 2;
        result.extend_from_slice(&field.data[start..start + count]);
    }
    result
}

impl LiquifyAdjustmentHistory {
    pub fn begin(&mut self) {
        self.pending.clear();
        self.pending_bytes = 0;
        self.limited = false;
    }
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    pub fn capture(
        &mut self,
        field: &crate::ops::transform::DisplacementField,
        point: [f32; 2],
        radius: f32,
    ) {
        if self.limited || field.width == 0 || field.height == 0 {
            return;
        }
        let x0 = (point[0] - radius).max(0.) as u32 / FIELD_TILE;
        let y0 = (point[1] - radius).max(0.) as u32 / FIELD_TILE;
        let x1 = ((point[0] + radius).ceil().max(0.) as u32).min(field.width - 1) / FIELD_TILE;
        let y1 = ((point[1] + radius).ceil().max(0.) as u32).min(field.height - 1) / FIELD_TILE;
        for y in y0..=y1 {
            for x in x0..=x1 {
                if self.pending.contains_key(&(x, y)) {
                    continue;
                }
                let before = field_tile(field, x * FIELD_TILE, y * FIELD_TILE);
                let reserved = before.len() * 8 + std::mem::size_of::<FieldTileChange>();
                if self.pending_bytes + reserved > self.limit {
                    self.pending.clear();
                    self.undo.clear();
                    self.redo.clear();
                    self.pending_bytes = 0;
                    self.stored_bytes = 0;
                    self.limited = true;
                    return;
                }
                while self.stored_bytes + self.pending_bytes + reserved > self.limit {
                    if !self.undo.is_empty() {
                        self.stored_bytes -= field_changes_bytes(&self.undo.remove(0));
                    } else if let Some(changes) = self.redo.pop() {
                        self.stored_bytes -= field_changes_bytes(&changes);
                    } else {
                        break;
                    }
                }
                self.pending_bytes += reserved;
                self.pending.insert((x, y), before);
            }
        }
    }
    pub fn finish(&mut self, field: &crate::ops::transform::DisplacementField) {
        if self.limited {
            return;
        }
        let mut changes = Vec::new();
        for ((x, y), before) in std::mem::take(&mut self.pending) {
            let x = x * FIELD_TILE;
            let y = y * FIELD_TILE;
            let after = field_tile(field, x, y);
            if before != after {
                changes.push(FieldTileChange {
                    x,
                    y,
                    before,
                    after,
                });
            }
        }
        self.pending_bytes = 0;
        if changes.is_empty() {
            return;
        }
        self.stored_bytes -= self
            .redo
            .iter()
            .map(|changes| field_changes_bytes(changes))
            .sum::<usize>();
        self.redo.clear();
        self.stored_bytes += field_changes_bytes(&changes);
        self.undo.push(changes);
    }
    fn restore(
        field: &mut crate::ops::transform::DisplacementField,
        changes: &[FieldTileChange],
        after: bool,
    ) {
        for tile in changes {
            let values = if after { &tile.after } else { &tile.before };
            let mut offset = 0;
            for row in tile.y..(tile.y + FIELD_TILE).min(field.height) {
                let start = (row as usize * field.width as usize + tile.x as usize) * 2;
                let count = ((tile.x + FIELD_TILE).min(field.width) - tile.x) as usize * 2;
                field.data[start..start + count].copy_from_slice(&values[offset..offset + count]);
                offset += count;
            }
        }
    }
    pub fn undo(&mut self, field: &mut crate::ops::transform::DisplacementField) -> bool {
        let Some(changes) = self.undo.pop() else {
            return false;
        };
        Self::restore(field, &changes, false);
        self.redo.push(changes);
        true
    }
    pub fn redo(&mut self, field: &mut crate::ops::transform::DisplacementField) -> bool {
        let Some(changes) = self.redo.pop() else {
            return false;
        };
        Self::restore(field, &changes, true);
        self.undo.push(changes);
        true
    }
}
