//! Small deterministic mutation run for local development and ordinary CI.
use paintfe_fuzz_targets::{check_expression, check_metadata};

fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("Expected an iteration count"))
        .unwrap_or(5000);
    assert!((1..=1_000_000).contains(&iterations));
    let seeds: &[&[u8]] = &[
        include_bytes!("../../../src/io/heic/max_pixels.txt"),
        include_bytes!("../../corpus/heif_metadata/alpha.heif"),
        b"(800 + 200) / 2",
        b"64 * 64 - 1",
        b"\0\0\0\x08free",
        b"(((1)))",
    ];
    let mut random = 0x83a1_75c9_u32;
    for index in 0..iterations {
        let mut bytes = seeds[index % seeds.len()].to_vec();
        for _ in 0..1 + index % 8 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            match random % 4 {
                0 if !bytes.is_empty() => {
                    let at = random as usize % bytes.len();
                    bytes[at] ^= (random >> 8) as u8;
                }
                1 if !bytes.is_empty() => bytes.truncate(random as usize % bytes.len()),
                2 if bytes.len() < 4096 => bytes.push((random >> 8) as u8),
                _ => bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]),
            }
        }
        check_metadata(&bytes);
        check_expression(&bytes);
    }
    println!("Completed {iterations} deterministic mutations through both production parsers.");
}
