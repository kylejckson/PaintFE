"""Local-only experiments. Requires Pillow and pillow-heif in an isolated venv.

Inputs and outputs stay under the explicitly supplied scratch directory.
No downloads, application integration, or repository changes are performed here.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw
import pillow_heif


def invoke(executable, *args):
    start = time.perf_counter()
    try:
        process = subprocess.run(
            [str(executable), *map(str, args)], capture_output=True,
            text=True, timeout=120, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_ms": 120000}
    return {
        "status": "ok" if process.returncode == 0 else "error",
        "exit": process.returncode,
        "wall_ms": round((time.perf_counter() - start) * 1000, 3),
        "stdout": process.stdout.strip(), "stderr": process.stderr.strip(),
    }


def differences(actual, reference):
    if len(actual) != len(reference):
        return {"length_mismatch": [len(actual), len(reference)]}
    # Sample at most 100k pixels; ignore invisible RGB, but always compare alpha.
    step = max(1, len(actual) // (4 * 100000)) * 4
    rgb, alpha = [], []
    for i in range(0, len(actual), step):
        alpha.append(abs(actual[i + 3] - reference[i + 3]))
        if reference[i + 3] > 0:
            rgb.extend(abs(actual[i + c] - reference[i + c]) for c in range(3))
    return {
        "sampled_rgb_mae": round(statistics.mean(rgb), 4) if rgb else 0,
        "sampled_rgb_max": max(rgb, default=0),
        "sampled_rgb_psnr_db": round(10 * math.log10(255**2 / statistics.mean(v*v for v in rgb)), 3)
        if any(rgb) else "exact",
        "sampled_alpha_mae": round(statistics.mean(alpha), 4),
        "sampled_alpha_max": max(alpha, default=0),
    }


def reference_heif(path):
    return pillow_heif.open_heif(path, convert_hdr_to_8bit=True).to_pillow().convert("RGBA")



def make_artwork():
    gradient = Image.linear_gradient("L").resize((512, 512))
    image = Image.merge("RGBA", (gradient, gradient.transpose(Image.Transpose.ROTATE_90), Image.new("L", (512, 512), 180), gradient))
    draw = ImageDraw.Draw(image)
    for x in range(0, 512, 8):
        draw.line((x, 0, 511 - x, 511), fill=(255, 0, 0, 255), width=1)
    draw.rectangle((80, 160, 430, 310), fill=(0, 0, 0, 255))
    draw.text((100, 200), "PaintFE AVIF: edges + alpha", fill=(255, 255, 255, 255))
    return image


def avif_samples(root, results):
    executable = (root / ("avif.exe" if os.name == "nt" else "avif")).resolve()
    fixtures = [("artwork", make_artwork())]
    photo = reference_heif(root / "samples/image_large.heic")
    photo.thumbnail((1920, 1080))
    fixtures.append(("photo", photo))
    for name, image in fixtures:
        raw = root / f"{name}.rgba"
        image.save(root / f"{name}.png")
        reference = image.tobytes()
        raw.write_bytes(reference)
        for speed, quality in [(8, 80), (4, 80), (8, 100)]:
            avif = root / f"{name}-s{speed}-q{quality}.avif"
            row = {"file": avif.name, "size": list(image.size), "speed": speed, "quality": quality}
            row["encode"] = invoke(executable, "avif-encode", raw, *image.size, avif, speed, quality, 3)
            if row["encode"]["status"] == "ok":
                row["encoded_bytes"] = avif.stat().st_size
                out = root / "avif-decoded.rgba"
                row["decode"] = invoke(executable, "avif-decode", avif, out, 5)
                with Image.open(avif) as decoded:
                    independent = decoded.convert("RGBA").tobytes()
                row["roundtrip"] = differences(independent, reference)
                if row["decode"]["status"] == "ok":
                    row["decoder_vs_pillow"] = differences(out.read_bytes(), independent)
            results["avif"].append(row)
            print(json.dumps(row), flush=True)
    bad = root / "invalid.avif"
    bad.write_bytes(b"not an avif")
    results["avif_invalid"] = invoke(executable, "avif-decode", bad, root / "invalid.rgba", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--phase", choices=("avif",), default="avif")
    args = parser.parse_args()
    root = args.scratch.resolve()
    report = root / f"results-{args.phase}.json"
    results = {"avif": [], "reference": pillow_heif.libheif_info()}
    avif_samples(root, results)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
