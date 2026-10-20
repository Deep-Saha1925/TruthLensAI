"""Group-aware dataset splitting.

Splitting by FILE leaks: frames of one video, edits of one photo, or a fake and its source
image can land in both train and test, and then the test score measures memorization.
This script splits by GROUP, so everything that shares a group id stays in one split.

    python -m training.split_dataset --source raw_data --out dataset --group-by prefix

Input:   source/real/**  and  source/fake/**   (any depth)
Output:  out/{train,validation,test}/{real,fake}/  +  split_manifest.csv  +  split_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from training.config import CLASS_NAMES, IMAGE_EXTENSIONS, SPLITS

MANIFEST_NAME = "split_manifest.csv"
REPORT_NAME = "split_report.json"
MANIFEST_FIELDS = ["split", "label", "group", "file", "source", "sha256", "format", "width", "height"]
DEFAULT_GROUP_REGEX = r"^(.*?)__"


class SplitError(Exception):
    pass


@dataclass
class Item:
    path: Path
    label: str
    group: str = ""
    sha256: str = ""
    fmt: str = ""
    width: int = 0
    height: int = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_source(source: Path) -> list[Item]:
    if not source.is_dir():
        raise SplitError(f"Source directory not found: {source}")
    items: list[Item] = []
    for label in CLASS_NAMES:
        folder = source / label
        if not folder.is_dir():
            raise SplitError(f"Expected a '{label}' folder inside {source}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(Item(path=path, label=label))
    if not items:
        raise SplitError("No images found in the source folders.")
    return items


def derive_group(item: Item, source: Path, method: str, pattern: str | None) -> str:
    """The group a file belongs to. The same id in real/ and fake/ means the same group."""
    rel = item.path.relative_to(source / item.label)
    own = f"file:{item.label}/{rel.as_posix()}"
    if method == "file":
        return own
    if method == "parent":
        parts = rel.parts[:-1]
        return "/".join(parts) if parts else own  # a loose file is its own group
    if method == "prefix":
        match = re.match(pattern or DEFAULT_GROUP_REGEX, rel.stem)
        if match is None:
            raise SplitError(
                f"File name '{rel.name}' does not match --group-regex. Rename the files, "
                "change the regex, or use --group-by parent or file."
            )
        group = match.group(1) if match.groups() else match.group(0)
        if not group:
            raise SplitError(f"--group-regex produced an empty group for '{rel.name}'.")
        return group
    raise SplitError(f"Unknown --group-by '{method}'.")


def inspect_item(item: Item) -> bool:
    """Read format/size, check the file is a valid image, and hash it. False if unreadable."""
    try:
        with Image.open(item.path) as image:
            item.fmt = (image.format or "").lower()
            item.width, item.height = image.size
            image.verify()
        item.sha256 = file_sha256(item.path)
        return True
    except Exception:
        return False


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a: str, b: str) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_b] = root_a
        return True


def merge_identical_content(items: list[Item], groups: _UnionFind) -> tuple[int, int]:
    """Files with identical bytes force their groups together. Returns (merges, label conflicts)."""
    by_hash: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_hash[item.sha256].append(item)
    merges = conflicts = 0
    for same in by_hash.values():
        if len(same) < 2:
            continue
        if len({item.label for item in same}) > 1:
            conflicts += 1  # the same image labelled both real and fake
        for other in same[1:]:
            merges += groups.union(same[0].group, other.group)
    return merges, conflicts


def average_hash(path: Path) -> np.ndarray:
    """64-bit perceptual hash as 8 bytes."""
    with Image.open(path) as image:
        small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    return np.packbits((pixels > pixels.mean()).flatten())


def find_near_duplicate_pairs(hashes: np.ndarray, max_distance: int, chunk: int = 64) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start : start + chunk]
        distance = np.unpackbits(block[:, None, :] ^ hashes[None, :, :], axis=2).sum(axis=2)
        rows, cols = np.nonzero(distance <= max_distance)
        for row, col in zip(rows, cols):
            i = start + int(row)
            if int(col) > i:
                pairs.append((i, int(col)))
    return pairs


def assign_groups(
    group_counts: dict[str, dict[str, int]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Greedy, seeded and deterministic: biggest groups first, each to the split it fills least.

    Only the classes a group actually contains count toward its cost, so a one-class group
    (the usual case: a real video, a fake video) balances its own class without being
    distorted by how full the splits are with the other class.
    """
    rng = random.Random(seed)
    groups = list(group_counts)
    rng.shuffle(groups)  # breaks ties between equally sized groups
    groups.sort(key=lambda g: -sum(group_counts[g].values()))

    totals = {label: sum(c.get(label, 0) for c in group_counts.values()) for label in CLASS_NAMES}
    assigned = {split: {label: 0 for label in CLASS_NAMES} for split in ratios}
    result: dict[str, str] = {}
    for group in groups:
        counts = group_counts[group]
        present = [label for label in CLASS_NAMES if counts.get(label, 0) > 0]
        best_split, best_cost = None, math.inf
        for split, ratio in ratios.items():
            cost = sum(
                (assigned[split][label] + counts[label]) / (ratio * totals[label])
                for label in present
            )
            if cost < best_cost:
                best_split, best_cost = split, cost
        result[group] = best_split  # type: ignore[assignment]
        for label in present:
            assigned[best_split][label] += counts[label]  # type: ignore[index]
    return result

def audit_shortcuts(items: list[Item]) -> list[str]:
    """Warn when the classes differ in ways a model could learn instead of the content."""
    warnings: list[str] = []
    formats = {label: Counter(i.fmt for i in items if i.label == label) for label in CLASS_NAMES}
    totals = {label: sum(counts.values()) for label, counts in formats.items()}
    if not all(totals.values()):
        return warnings

    names = set().union(*formats.values())
    distance = 0.5 * sum(
        abs(formats["real"][n] / totals["real"] - formats["fake"][n] / totals["fake"]) for n in names
    )
    if distance > 0.5:
        warnings.append(
            "The classes differ strongly in file format "
            f"(real: {dict(formats['real'])}, fake: {dict(formats['fake'])}). A model can learn "
            "the format instead of the content. Re-encode both classes the same way."
        )
    megapixels = {
        label: float(np.median([i.width * i.height / 1e6 for i in items if i.label == label]))
        for label in CLASS_NAMES
    }
    low, high = sorted(megapixels.values())
    if low > 0 and high / low > 2:
        warnings.append(
            f"Median image size differs by more than 2x between classes ({megapixels}). A model "
            "can learn the size. Resize or crop both classes consistently."
        )
    return warnings


def _ratio_map(ratios) -> dict[str, float]:
    if len(ratios) != 3 or any(r <= 0 for r in ratios):
        raise SplitError("--ratios needs three positive numbers (train validation test).")
    total = float(sum(ratios))
    return {split: ratio / total for split, ratio in zip(SPLITS, ratios)}


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "group"


def _destination(folder: Path, group: str, sha: str, suffix: str) -> Path:
    base = f"{_safe(group)[:60]}__{sha[:10]}"
    candidate = folder / f"{base}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = folder / f"{base}-{counter}{suffix}"
        counter += 1
    return candidate


def _place(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass  # different volume or unsupported: fall back to a copy
    shutil.copy2(source, destination)


def split_dataset(
    source,
    out,
    *,
    group_by: str = "parent",
    group_regex: str | None = None,
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
    mode: str = "copy",
    near_distance: int = -1,
    near_max_images: int = 20000,
) -> dict:
    source, out = Path(source), Path(out)
    ratio_map = _ratio_map(ratios)
    if out.exists() and any(out.iterdir()):
        raise SplitError(f"Output directory {out} is not empty. Choose a new --out so runs never mix.")

    items = discover_source(source)
    usable = [item for item in items if inspect_item(item)]
    unreadable = len(items) - len(usable)
    if not usable:
        raise SplitError("No readable images.")

    for item in usable:
        item.group = derive_group(item, source, group_by, group_regex)
    groups = _UnionFind()
    for item in usable:
        groups.find(item.group)

    warnings: list[str] = []
    identical_merges, label_conflicts = merge_identical_content(usable, groups)
    if label_conflicts:
        warnings.append(
            f"{label_conflicts} image(s) appear under both real/ and fake/ (identical bytes). "
            "That is label noise; remove them."
        )

    near_merges = 0
    if near_distance >= 0:
        if len(usable) > near_max_images:
            warnings.append(
                f"Near-duplicate check skipped: {len(usable)} images exceed --near-dup-max-images "
                f"({near_max_images})."
            )
        else:
            hashed, vectors = [], []
            for item in usable:
                try:
                    vectors.append(average_hash(item.path))
                    hashed.append(item)
                except Exception:
                    continue
            if len(hashed) > 1:
                for i, j in find_near_duplicate_pairs(np.stack(vectors), near_distance):
                    near_merges += groups.union(hashed[i].group, hashed[j].group)

    for item in usable:
        item.group = groups.find(item.group)

    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in usable:
        group_counts[item.group][item.label] += 1
    assignment = assign_groups(group_counts, ratio_map, seed)

    placed = {split: Counter() for split in SPLITS}
    for item in usable:
        placed[assignment[item.group]][item.label] += 1
    for split in SPLITS:
        for label in CLASS_NAMES:
            if placed[split][label] == 0:
                raise SplitError(
                    f"The {split} split would contain no {label} images. Each split needs at least "
                    "one group per class: add data, use more (smaller) groups, or adjust --ratios."
                )

    largest = max(sum(counts.values()) for counts in group_counts.values())
    if largest / len(usable) > 0.25:
        warnings.append(
            f"The largest group holds {largest / len(usable):.0%} of all images, so the splits "
            "cannot match the requested ratios closely."
        )
    warnings.extend(audit_shortcuts(usable))

    rows = []
    for item in usable:
        split = assignment[item.group]
        destination = _destination(out / split / item.label, item.group, item.sha256, item.path.suffix.lower())
        _place(item.path, destination, mode)
        rows.append({
            "split": split, "label": item.label, "group": item.group,
            "file": destination.relative_to(out).as_posix(), "source": str(item.path),
            "sha256": item.sha256, "format": item.fmt, "width": item.width, "height": item.height,
        })

    with (out / MANIFEST_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "total_images": len(usable),
        "unreadable_skipped": unreadable,
        "group_by": group_by,
        "ratios": ratio_map,
        "seed": seed,
        "counts": {split: dict(placed[split]) for split in SPLITS},
        "groups_per_split": {
            split: sum(1 for g, s in assignment.items() if s == split) for split in SPLITS
        },
        "groups_merged_by_identical_content": identical_merges,
        "groups_merged_by_near_duplicates": near_merges,
        "label_conflicts": label_conflicts,
        "warnings": warnings,
        "manifest_sha256": file_sha256(out / MANIFEST_NAME),
    }
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def check_manifest_integrity(data_dir) -> list[str]:
    """Problems that would invalidate evaluation: leaking groups, missing or untracked files."""
    data_dir = Path(data_dir)
    manifest = data_dir / MANIFEST_NAME
    try:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return [f"cannot read {MANIFEST_NAME}: {exc}"]

    problems: list[str] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    missing = misplaced = 0
    for row in rows:
        group_splits[row["group"]].add(row["split"])
        if not (data_dir / row["file"]).is_file():
            missing += 1
        if not row["file"].startswith(f"{row['split']}/{row['label']}/"):
            misplaced += 1

    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        problems.append(
            f"{len(leaking)} group(s) appear in more than one split (for example {leaking[0]!r})"
        )
    if missing:
        problems.append(f"{missing} file(s) listed in the manifest are missing")
    if misplaced:
        problems.append(f"{misplaced} manifest row(s) point to a different split/class folder")

    on_disk = sum(
        1
        for split in SPLITS
        for label in CLASS_NAMES
        if (data_dir / split / label).is_dir()
        for path in (data_dir / split / label).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if on_disk != len(rows):
        problems.append(
            f"{on_disk} images on disk but {len(rows)} in the manifest "
            "(files added or removed after splitting?)"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group-aware train/validation/test split.")
    parser.add_argument("--source", required=True, type=Path, help="Folder with real/ and fake/ subfolders")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    parser.add_argument("--group-by", choices=("parent", "prefix", "file"), default="parent")
    parser.add_argument("--group-regex", default=None, help=f"For --group-by prefix (default {DEFAULT_GROUP_REGEX})")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--near-duplicate-distance", type=int, default=-1,
                        help="Merge groups whose images differ by at most N hash bits (-1 = off, try 4)")
    parser.add_argument("--near-dup-max-images", type=int, default=20000)
    args = parser.parse_args(argv)

    try:
        report = split_dataset(
            args.source, args.out, group_by=args.group_by, group_regex=args.group_regex,
            ratios=args.ratios, seed=args.seed, mode=args.mode,
            near_distance=args.near_duplicate_distance, near_max_images=args.near_dup_max_images,
        )
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Split {report['total_images']} images into {args.out}")
    for split in SPLITS:
        counts = report["counts"][split]
        print(f"  {split:<11} real {counts.get('real', 0):>6}  fake {counts.get('fake', 0):>6}  "
              f"groups {report['groups_per_split'][split]}")
    print(f"Merged by identical content: {report['groups_merged_by_identical_content']}, "
          f"by near-duplicates: {report['groups_merged_by_near_duplicates']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())