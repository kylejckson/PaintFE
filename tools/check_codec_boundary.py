"""Keep the obsolete Heifer HEVC experiment out of all shipped dependency graphs."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BANNED = {"heifer", "heifer-isobmff", "heifer-hevc-dec", "heifer-hevc-enc"}
PACKAGES = (".", "web", "tools/codec_probe", "optional/jxl-encoder", "optional/avif-codec")


def check(root=ROOT):
    errors = []
    for retired in ("optional/heic-encoder", "tools/build_heic_native.py"):
        if (root / retired).exists():
            errors.append(f"{retired}: retired codec build machinery")
    for package in PACKAGES:
        directory = root / package
        for filename in ("Cargo.toml", "Cargo.lock"):
            path = directory / filename
            if not path.exists():
                continue
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
            if filename.endswith("lock"):
                names = {row["name"] for row in parsed.get("package", [])}
            else:
                names = set()

                def dependencies(table):
                    for key, value in table.items():
                        if isinstance(value, dict):
                            if key in ("dependencies", "dev-dependencies", "build-dependencies") or key == "crates-io":
                                for alias, dependency in value.items():
                                    names.add(dependency.get("package", alias) if isinstance(dependency, dict) else alias)
                            dependencies(value)
                dependencies(parsed)
            for name in sorted(names & BANNED):
                errors.append(f"{path.relative_to(root)}: obsolete codec {name}")
    return errors


if __name__ == "__main__":
    failures = check()
    print("\n".join(failures) if failures else "Codec boundary passed: Heifer dependencies and retired HEVC build machinery absent.")
    raise SystemExit(bool(failures))
