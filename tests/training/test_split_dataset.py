import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.split_dataset import (
    Item,
    SplitError,
    assign_groups,
    check_manifest_integrity,
    derive_group,
    split_dataset,
)


def png(path: Path, seed: int, size: int = 40) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def build_source(root: Path, groups: int = 12, per_group: int = 3) -> Path:
    seed = 0
    for label, prefix in (("real", "r"), ("fake", "f")):
        for group in range(groups):
            for k in range(per_group):
                png(root / label / f"{prefix}{group:03d}__{k}.png", seed)
                seed += 1
    return root


def read_manifest(out: Path) -> list[dict]:
    with (out / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_for(rows: list[dict], name: str) -> dict:
    return next(row for row in rows if row["source"].endswith(name))


def split(source, out, **kwargs):
    return split_dataset(source, out, group_by="prefix", group_regex=r"^(.*?)__", **kwargs)


# ---- grouping ----------------------------------------------------------------------------


def test_group_by_prefix_parent_and_file(tmp_path):
    source = tmp_path
    item = Item(path=source / "real" / "vid1__f0003.png", label="real")
    assert derive_group(item, source, "prefix", r"^(.*?)__") == "vid1"

    nested = Item(path=source / "real" / "video1" / "frame.png", label="real")
    assert derive_group(nested, source, "parent", None) == "video1"

    loose = Item(path=source / "real" / "solo.png", label="real")
    assert derive_group(loose, source, "parent", None) == "file:real/solo.png"
    assert derive_group(loose, source, "file", None) == "file:real/solo.png"


def test_file_name_without_the_separator_is_an_error(tmp_path):
    item = Item(path=tmp_path / "real" / "nounderscore.png", label="real")
    with pytest.raises(SplitError):
        derive_group(item, tmp_path, "prefix", None)


# ---- assignment --------------------------------------------------------------------------


def test_assignment_keeps_groups_whole_balances_classes_and_is_deterministic():
    counts = {f"r{i}": {"real": 5} for i in range(40)} | {f"f{i}": {"fake": 5} for i in range(40)}
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}

    first = assign_groups(counts, ratios, seed=3)
    assert first == assign_groups(counts, ratios, seed=3)
    assert set(first) == set(counts)  # every group assigned exactly once

    for label, prefix in (("real", "r"), ("fake", "f")):
        per_split = {s: sum(5 for g, split_ in first.items() if split_ == s and g.startswith(prefix))
                     for s in ratios}
        assert abs(per_split["train"] - 140) <= 5
        assert abs(per_split["validation"] - 30) <= 5
        assert abs(per_split["test"] - 30) <= 5


# ---- end to end --------------------------------------------------------------------------


def test_no_group_crosses_splits_and_every_split_has_both_classes(split_data):
    rows = read_manifest(split_data)
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    for split_name in ("train", "validation", "test"):
        labels = {row["label"] for row in rows if row["split"] == split_name}
        assert labels == {"real", "fake"}
    assert check_manifest_integrity(split_data) == []
    assert (split_data / "split_report.json").is_file()


def test_identical_files_in_different_groups_are_kept_together(tmp_path):
    source = build_source(tmp_path / "raw")
    shutil.copy2(source / "real" / "r000__0.png", source / "real" / "r005__0.png")
    report = split(source, tmp_path / "out", seed=2)

    rows = read_manifest(tmp_path / "out")
    first, second = row_for(rows, "r000__0.png"), row_for(rows, "r005__0.png")
    assert first["split"] == second["split"]
    assert first["group"] == second["group"]
    assert report["groups_merged_by_identical_content"] >= 1


def test_the_same_image_labelled_real_and_fake_is_reported(tmp_path):
    source = build_source(tmp_path / "raw")
    shutil.copy2(source / "real" / "r001__0.png", source / "fake" / "f001__0.png")
    report = split(source, tmp_path / "out")
    assert report["label_conflicts"] == 1
    assert any("both real/ and fake/" in warning for warning in report["warnings"])


def test_near_duplicates_are_merged_when_requested(tmp_path):
    source = build_source(tmp_path / "raw")
    base = png(source / "real" / "r000__9.png", 999)
    nudged = np.clip(base.astype(int) + 1, 0, 255).astype(np.uint8)
    Image.fromarray(nudged).save(source / "real" / "r007__9.png")

    report = split(source, tmp_path / "out", near_distance=2)
    rows = read_manifest(tmp_path / "out")
    assert row_for(rows, "r000__9.png")["split"] == row_for(rows, "r007__9.png")["split"]
    assert report["groups_merged_by_near_duplicates"] >= 1


def test_format_shortcut_between_classes_is_flagged(tmp_path):
    source = tmp_path / "raw"
    for g in range(12):
        for k in range(2):
            png(source / "real" / f"r{g:03d}__{k}.jpg", 10 * g + k)
            png(source / "fake" / f"f{g:03d}__{k}.png", 500 + 10 * g + k)
    report = split(source, tmp_path / "out")
    assert any("format" in warning for warning in report["warnings"])


def test_an_existing_output_directory_is_refused(tmp_path):
    source = build_source(tmp_path / "raw")
    out = tmp_path / "out"
    out.mkdir()
    (out / "something.txt").write_text("x")
    with pytest.raises(SplitError):
        split(source, out)


def test_too_few_groups_cannot_fill_every_split(tmp_path):
    source = build_source(tmp_path / "raw", groups=2)
    with pytest.raises(SplitError) as caught:
        split(source, tmp_path / "out")
    assert "would contain no" in str(caught.value)


# ---- manifest integrity ------------------------------------------------------------------


def rewrite_manifest(data_dir: Path, change) -> None:
    rows = read_manifest(data_dir)
    change(rows)
    with (data_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_leaking_manifest_is_detected(split_data, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(split_data, copy)

    def leak(rows):
        train_row = next(r for r in rows if r["split"] == "train")
        train_row["split"] = "test"

    rewrite_manifest(copy, leak)
    assert any("more than one split" in problem for problem in check_manifest_integrity(copy))


def test_missing_and_untracked_files_are_detected(split_data, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(split_data, copy)
    (copy / next(r["file"] for r in read_manifest(copy))).unlink()
    problems = check_manifest_integrity(copy)
    assert any("missing" in problem for problem in problems)
    assert any("on disk" in problem for problem in problems)

def test_paired_groups_balance_both_classes():
    counts = {f"p{i}": {"real": 2, "fake": 2} for i in range(40)}
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    assignment = assign_groups(counts, ratios, seed=1)
    for label in ("real", "fake"):
        per_split = {s: sum(2 for g, split_ in assignment.items() if split_ == s) for s in ratios}
        assert abs(per_split["train"] - 56) <= 2
        assert abs(per_split["validation"] - 12) <= 2
        assert abs(per_split["test"] - 12) <= 2