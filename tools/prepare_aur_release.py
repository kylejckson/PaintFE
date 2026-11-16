"""Verify an owner-published release before touching trusted AUR recipes."""
import json
import os
import re
import subprocess
from pathlib import Path

REPOSITORY = "kylejckson/PaintFE"
RECIPES = ("PKGBUILD", "PKGBUILD-bin")


def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True,
                                             encoding="utf-8", timeout=60))


def verified_version(version, repository, event, expected_sha="", request=api):
    if repository != REPOSITORY or event not in ("push", "workflow_dispatch"):
        raise ValueError("AUR publishing requires an owner release or manual rerun")
    if not re.fullmatch(r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("Expected a numeric published version, for example 1.5.1")
    version = version.removeprefix("v")
    tag = "v" + version
    prefix = "repos/" + REPOSITORY
    release = request(prefix + "/releases/tags/" + tag)
    if release.get("tag_name") != tag or release.get("draft", True) or not release.get("published_at"):
        raise ValueError("Release is not published")
    assets = {asset["name"] for asset in release.get("assets", [])}
    if not {"PaintFE-linux-x64.tar.gz", "checksums-SHA256.txt"} <= assets:
        raise ValueError("Release is missing required Linux downloads")
    commit = request(prefix + "/commits/" + tag).get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid release commit")
    if event == "push" and (not expected_sha or expected_sha != commit):
        raise ValueError("Release tag does not match the invoking commit")
    main = request(prefix + "/git/ref/heads/main")["object"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", main):
        raise ValueError("Invalid main commit")
    comparison = request(prefix + "/compare/" + commit + "..." + main)
    if comparison.get("status") not in ("ahead", "identical"):
        raise ValueError("Release commit is not on main")
    return version


def prepare(root, version):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Invalid version")
    destination = root / "target" / "aur"
    destination.mkdir(parents=True, exist_ok=True)
    for name in RECIPES:
        source = (root / "packaging" / "aur" / name).read_text(encoding="utf-8")
        text, replacements = re.subn(r"^pkgver=.*$", "pkgver=" + version, source, flags=re.M)
        if replacements != 1:
            raise ValueError("Expected one pkgver in " + name)
        (destination / name).write_text(text, encoding="utf-8")


def main():
    version = verified_version(os.environ["AUR_VERSION"], os.environ["AUR_REPOSITORY"],
                               os.environ["AUR_EVENT"], os.environ.get("AUR_RELEASE_SHA", ""))
    prepare(Path(__file__).resolve().parents[1], version)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write("version=" + version + "\n")
    print("Prepared AUR recipes for verified release v" + version)


if __name__ == "__main__":
    main()
