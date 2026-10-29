//! Bounded, deterministic RGBA palette generation and optional error diffusion.
use image::RgbaImage;
use std::collections::BTreeMap;

struct SearchNode {
    point: [f32; 4],
    index: usize,
    axis: usize,
    left: Option<Box<SearchNode>>,
    right: Option<Box<SearchNode>>,
}
impl SearchNode {
    fn build(points: &mut [(usize, [f32; 4])]) -> Option<Box<Self>> {
        if points.is_empty() {
            return None;
        }
        let axis = (0..4)
            .max_by(|&a, &b| {
                let range = |c| {
                    points
                        .iter()
                        .map(|p| p.1[c])
                        .fold(f32::NEG_INFINITY, f32::max)
                        - points.iter().map(|p| p.1[c]).fold(f32::INFINITY, f32::min)
                };
                range(a).total_cmp(&range(b))
            })
            .unwrap();
        points.sort_unstable_by(|a, b| a.1[axis].total_cmp(&b.1[axis]));
        let middle = points.len() / 2;
        let (index, point) = points[middle];
        let (left, right) = points.split_at_mut(middle);
        Some(Box::new(Self {
            point,
            index,
            axis,
            left: Self::build(left),
            right: Self::build(&mut right[1..]),
        }))
    }
    fn find(&self, point: &[f32; 4], best: &mut (f32, usize)) {
        let distance = (0..4)
            .map(|c| (self.point[c] - point[c]).powi(2))
            .sum::<f32>();
        if distance < best.0 || (distance == best.0 && self.index < best.1) {
            *best = (distance, self.index);
        }
        let difference = point[self.axis] - self.point[self.axis];
        let (near, far) = if difference < 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };
        if let Some(node) = near {
            node.find(point, best);
        }
        if difference * difference <= best.0
            && let Some(node) = far
        {
            node.find(point, best);
        }
    }
}

pub fn quantize(image: &RgbaImage, colors: usize, dither: bool) -> (Vec<[u8; 4]>, Vec<u8>) {
    let colors = colors.clamp(1, 256);
    let mut exact = BTreeMap::new();
    for pixel in image.pixels() {
        let next = exact.len() as u8;
        exact.entry(pixel.0).or_insert(next);
        if exact.len() > colors {
            break;
        }
    }
    if exact.len() <= colors {
        let mut palette = vec![[0; 4]; exact.len()];
        for (&color, &index) in &exact {
            palette[index as usize] = color;
        }
        return (palette, image.pixels().map(|p| exact[&p.0]).collect());
    }
    let stride = (image.len() / 4).div_ceil(65_536).max(1);
    let sample: Vec<[u8; 4]> = image.pixels().step_by(stride).map(|p| p.0).collect();
    let mut boxes = vec![sample];
    while boxes.len() < colors {
        let choice = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let ranges: [u8; 4] = std::array::from_fn(|c| {
                    b.iter().map(|p| p[c]).max().unwrap() - b.iter().map(|p| p[c]).min().unwrap()
                });
                let channel = (0..4)
                    .max_by_key(|&c| u16::from(ranges[c]) * if c == 3 { 2 } else { 1 })
                    .unwrap();
                (i, channel, ranges[channel])
            })
            .max_by_key(|&(i, _, range)| usize::from(range) * boxes[i].len());
        let Some((index, channel, range)) = choice else {
            break;
        };
        if range == 0 {
            break;
        }
        boxes[index].sort_unstable_by_key(|p| p[channel]);
        let middle = boxes[index].len() / 2;
        let rest = boxes[index].split_off(middle);
        boxes.push(rest);
    }
    let palette: Vec<[u8; 4]> = boxes
        .iter()
        .map(|b| {
            std::array::from_fn(|c| {
                ((b.iter().map(|p| u64::from(p[c])).sum::<u64>() + b.len() as u64 / 2)
                    / b.len() as u64) as u8
            })
        })
        .collect();
    let width = image.width() as usize;
    let mut points: Vec<_> = palette
        .iter()
        .enumerate()
        .map(|(index, p)| {
            (
                index,
                [
                    f32::from(p[0]),
                    f32::from(p[1]),
                    f32::from(p[2]),
                    f32::from(p[3]) * 2.0,
                ],
            )
        })
        .collect();
    let search = SearchNode::build(&mut points).unwrap();
    let mut errors = vec![[0.0f32; 3]; width + 2];
    let mut next = errors.clone();
    let mut indices = Vec::with_capacity(image.len() / 4);
    for y in 0..image.height() {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y).0;
            let adjusted: [f32; 4] = std::array::from_fn(|c| {
                (f32::from(pixel[c])
                    + if dither && c < 3 && pixel[3] > 0 {
                        errors[x + 1][c]
                    } else {
                        0.0
                    })
                .clamp(0.0, 255.0)
            });
            let mut best = (f32::INFINITY, usize::MAX);
            search.find(
                &[adjusted[0], adjusted[1], adjusted[2], adjusted[3] * 2.0],
                &mut best,
            );
            let index = best.1;
            indices.push(index as u8);
            if dither && pixel[3] > 0 {
                for (c, &value) in adjusted.iter().take(3).enumerate() {
                    let error = value - f32::from(palette[index][c]);
                    errors[x + 2][c] += error * 7.0 / 16.0;
                    next[x][c] += error * 3.0 / 16.0;
                    next[x + 1][c] += error * 5.0 / 16.0;
                    next[x + 2][c] += error / 16.0;
                }
            }
        }
        std::mem::swap(&mut errors, &mut next);
        next.fill([0.0; 3]);
    }
    (palette, indices)
}

pub fn gif(image: &RgbaImage, colors: usize, dither: bool) -> (Vec<u8>, Vec<u8>, Option<u8>) {
    let transparent = image.pixels().any(|p| p[3] < 128);
    let mut opaque = image.clone();
    for p in opaque.pixels_mut() {
        p[3] = 255;
    }
    let (mut palette, mut indices) = quantize(
        &opaque,
        colors.clamp(2, 256) - usize::from(transparent),
        dither,
    );
    let transparency = if transparent {
        let index = palette.len() as u8;
        palette.push([0, 0, 0, 0]);
        for (p, i) in image.pixels().zip(&mut indices) {
            if p[3] < 128 {
                *i = index;
            }
        }
        Some(index)
    } else {
        None
    };
    (
        palette
            .into_iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        indices,
        transparency,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn palette_search_matches_exhaustive_rgba_distance() {
        let mut points: Vec<_> = (0..127)
            .map(|i| {
                (
                    i,
                    [
                        (i * 17 % 256) as f32,
                        (i * 31 % 256) as f32,
                        (i * 47 % 256) as f32,
                        (i * 19 % 256) as f32 * 2.0,
                    ],
                )
            })
            .collect();
        let original = points.clone();
        let tree = SearchNode::build(&mut points).unwrap();
        for i in 0..400 {
            let point = [
                (i * 13 % 256) as f32,
                (i * 23 % 256) as f32,
                (i * 43 % 256) as f32,
                (i * 7 % 256) as f32 * 2.0,
            ];
            let expected = original
                .iter()
                .map(|(index, p)| {
                    (
                        (0..4).map(|c| (p[c] - point[c]).powi(2)).sum::<f32>(),
                        *index,
                    )
                })
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                .unwrap();
            let mut actual = (f32::INFINITY, usize::MAX);
            tree.find(&point, &mut actual);
            assert_eq!(actual, expected);
        }
    }
}
