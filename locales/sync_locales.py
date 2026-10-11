"""Utilities for keeping PaintFE locale files sorted and synchronized."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

LOCALES_DIR = Path(__file__).resolve().parent
ENGLISH_FILE = LOCALES_DIR / "en.txt"
CUSTOM_MENU_ORDER = {
    "menu.file": 0,
    "menu.edit": 1,
    "menu.canvas": 2,
    "menu.color": 3,
    "menu.filter": 4,
    "menu.generate": 5,
    "menu.view": 6,
}

SortComponent = Tuple[int, Union[int, str]]
EntryPathSortKey = Tuple[SortComponent, ...]
LocaleSortKey = Tuple[int, Tuple[int, ...], EntryPathSortKey]


@dataclass(frozen=True)
class LocaleEntry:
    """Represents one localization entry together with its leading comments."""

    key: str
    line: str
    prefix_lines: Tuple[str, ...]


@dataclass(frozen=True)
class LocaleFile:
    """Stores the parsed content and formatting metadata for a locale file."""

    path: Path
    entries: Tuple[LocaleEntry, ...]
    trailing_lines: Tuple[str, ...]
    newline: str
    text: str


@dataclass(frozen=True)
class SyncIssues:
    """Describes missing and extra keys discovered in a locale file."""

    missing_keys: Tuple[str, ...]
    extra_keys: Tuple[str, ...]


def detect_newline(raw: bytes) -> str:
    """Detect the dominant newline style used in a file."""

    if b"\r\n" in raw:
        return "\r\n"
    if b"\r" in raw:
        return "\r"
    return "\n"


def read_locale_file(path: Path) -> LocaleFile:
    """Read and validate a locale file while preserving comments and spacing."""

    raw = path.read_bytes()
    newline = detect_newline(raw)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: expected UTF-8 text") from exc

    entries: List[LocaleEntry] = []
    pending_prefix: List[str] = []
    seen_keys: Dict[str, int] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            pending_prefix.append(line)
            continue

        if "=" not in line:
            raise ValueError(f"{path.name}:{line_number}: expected key=value entry")

        key, _ = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path.name}:{line_number}: empty localization key")

        previous_line = seen_keys.get(key)
        if previous_line is not None:
            raise ValueError(
                f"{path.name}:{line_number}: duplicate key '{key}' "
                f"(first defined on line {previous_line})"
            )
        seen_keys[key] = line_number

        entries.append(
            LocaleEntry(
                key=key,
                line=line,
                prefix_lines=tuple(pending_prefix),
            )
        )
        pending_prefix.clear()

    return LocaleFile(
        path=path,
        entries=tuple(entries),
        trailing_lines=tuple(pending_prefix),
        newline=newline,
        text=text,
    )


def segment_sort_key(path_segments: List[str], index: int) -> SortComponent:
    """Build the sortable token for one key segment."""

    if index == 1 and path_segments[0] == "menu":
        path = ".".join(path_segments[: index + 1])
        custom_position = CUSTOM_MENU_ORDER.get(path)
        if custom_position is not None:
            return (0, custom_position)
    return (1, path_segments[index].casefold())


def entry_sort_key(entry: LocaleEntry) -> EntryPathSortKey:
    """Create the canonical sort key for an entry inside the reference locale."""

    path_segments = entry.key.split(".")
    menu_group = (0, 0) if entry.key == "menu" or entry.key.startswith("menu.") else (1, 0)
    return (menu_group,) + tuple(
        segment_sort_key(path_segments, index) for index in range(len(path_segments))
    )


def sort_reference_entries(entries: Tuple[LocaleEntry, ...]) -> Tuple[LocaleEntry, ...]:
    """Sort the English reference entries into the canonical project order."""

    return tuple(sorted(entries, key=entry_sort_key))


def locale_sort_key(
    entry: LocaleEntry, reference_positions: Dict[str, int]
) -> LocaleSortKey:
    """Create a sort key that follows English order for known keys."""

    fallback_key = entry_sort_key(entry)
    reference_position = reference_positions.get(entry.key)
    if reference_position is not None:
        return (0, (reference_position,), fallback_key)
    return (1, (0,), fallback_key)


def sort_locale_entries(
    entries: Tuple[LocaleEntry, ...], reference_positions: Dict[str, int]
) -> Tuple[LocaleEntry, ...]:
    """Sort a translated locale file to match the canonical English order."""

    return tuple(
        sorted(entries, key=lambda entry: locale_sort_key(entry, reference_positions))
    )


def render_locale_file(locale_file: LocaleFile, entries: Tuple[LocaleEntry, ...]) -> str:
    """Render a locale file without empty lines or trailing separators."""

    lines: List[str] = []
    for entry in entries:
        lines.extend(filter(None, entry.prefix_lines))
        lines.append(entry.line)
    lines.extend(filter(None, locale_file.trailing_lines))
    if not lines:
        return locale_file.newline
    return locale_file.newline.join(lines) + locale_file.newline


def write_if_changed(
    locale_file: LocaleFile, entries: Tuple[LocaleEntry, ...], check: bool
) -> bool:
    """Rewrite a locale file only when the sorted content differs."""

    rendered = render_locale_file(locale_file, entries)
    if locale_file.text == rendered:
        return False

    if check:
        return True

    with locale_file.path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(rendered)
    return True


def describe_change(path: Path, changed: bool, check: bool, num_keys: int) -> None:
    """Print whether a locale file changed or was already in sync."""

    action = "Would update" if check else "Updated"
    if changed:
        print(f"{action}: {path.name} ({num_keys} keys found)")
    else:
        print(f"Already sorted: {path.name} ({num_keys} keys found)")


def find_sync_issues(
    locale_file: LocaleFile, reference_keys: Tuple[str, ...], reference_positions: Dict[str, int]
) -> SyncIssues:
    """Compare one locale file against the English reference key set."""

    locale_keys = {entry.key for entry in locale_file.entries}
    missing_keys = tuple(key for key in reference_keys if key not in locale_keys)
    extra_entries = tuple(
        entry for entry in locale_file.entries if entry.key not in reference_positions
    )
    extra_keys = tuple(entry.key for entry in sort_reference_entries(extra_entries))
    return SyncIssues(missing_keys=missing_keys, extra_keys=extra_keys)


def report_sync_issues(
    path: Path,
    issues: SyncIssues,
    hide_missing_list: bool = False,
    hide_extra_list: bool = False,
) -> bool:
    """Print the missing and extra keys for one translated locale file."""

    print(f"Sync report: {path.name}")
    if not issues.missing_keys and not issues.extra_keys:
        print("  sync issues: none")
        return False

    if issues.missing_keys:
        if hide_missing_list:
            print(f"  missing keys count: {len(issues.missing_keys)}")
        else:
            print(f"  missing keys ({len(issues.missing_keys)}):")
            for key in issues.missing_keys:
                print(f"    - {key}")
    elif not hide_missing_list:
        print("  missing keys: none")

    if issues.extra_keys:
        if hide_extra_list:
            print(f"  extra keys count: {len(issues.extra_keys)}")
        else:
            print(f"  extra keys ({len(issues.extra_keys)}):")
            for key in issues.extra_keys:
                print(f"    - {key}")
    elif not hide_extra_list:
        print("  extra keys: none")

    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the locale synchronization script."""

    parser = argparse.ArgumentParser(
        description="Sort and validate PaintFE localization files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report pending changes without rewriting files",
    )
    parser.add_argument(
        "--hidemissinglist",
        action="store_true",
        help="show only the number of missing keys instead of the full list",
    )
    parser.add_argument(
        "--hideextralist",
        action="store_true",
        help="show only the number of extra keys instead of the full list",
    )
    return parser.parse_args()


def main() -> int:
    """Sort locale files, compare them to English, and print sync issues."""

    args = parse_args()

    locale_paths = sorted(LOCALES_DIR.glob("*.txt"))
    if not locale_paths:
        raise FileNotFoundError(f"No locale files found in {LOCALES_DIR}")

    locale_files = {path.name: read_locale_file(path) for path in locale_paths}
    english_file = locale_files.get(ENGLISH_FILE.name)
    if english_file is None:
        raise FileNotFoundError(f"Missing canonical locale file: {ENGLISH_FILE.name}")

    sorted_english_entries = sort_reference_entries(english_file.entries)
    reference_keys = tuple(entry.key for entry in sorted_english_entries)
    reference_positions = {key: index for index, key in enumerate(reference_keys)}

    any_changed = False
    changed = write_if_changed(english_file, sorted_english_entries, check=args.check)
    any_changed = any_changed or changed
    describe_change(english_file.path, changed, args.check, len(sorted_english_entries))

    any_sync_issues = False
    for path in locale_paths:
        if path.name == ENGLISH_FILE.name:
            continue

        locale_file = locale_files[path.name]
        sorted_entries = sort_locale_entries(locale_file.entries, reference_positions)
        changed = write_if_changed(locale_file, sorted_entries, check=args.check)
        any_changed = any_changed or changed
        describe_change(locale_file.path, changed, args.check, len(sorted_entries))

        issues = find_sync_issues(locale_file, reference_keys, reference_positions)
        any_sync_issues = (
            report_sync_issues(
                locale_file.path,
                issues,
                hide_missing_list=args.hidemissinglist,
                hide_extra_list=args.hideextralist,
            )
            or any_sync_issues
        )

    print(f"Validated locale files: {len(locale_files)}")
    if any_sync_issues:
        return 1
    if args.check and any_changed:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
