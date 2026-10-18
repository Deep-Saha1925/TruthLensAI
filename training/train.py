"""Train an image detector.  Run from the repository root:  python -m training.train --help"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.augmentations import Augmenter
from training.config import TrainConfig, build_parser, config_from_args
from training.dataset import (
    ImageDataset,
    class_counts,
    class_weights,
    discover_samples,
    sample_weights,
)
from training.metrics import json_safe, summarize_binary
from training.modeling import build_model, predict_probabilities, resolve_device, set_seed
from training.split_dataset import MANIFEST_NAME, check_manifest_integrity, file_sha256


class TrainingError(Exception):
    pass


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def _fmt(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.4f}"


def _check_split_integrity(data_dir: Path, require_manifest: bool) -> str | None:
    manifest = data_dir / MANIFEST_NAME
    if manifest.is_file():
        problems = check_manifest_integrity(data_dir)
        if problems:
            raise TrainingError(
                "Split integrity check failed (results would be invalid):\n - " + "\n - ".join(problems)
            )
        return file_sha256(manifest)
    if require_manifest:
        raise TrainingError(
            f"{manifest} not found. Split the data with training.split_dataset, or drop --require-manifest."
        )
    print(
        "WARNING: no split_manifest.csv. This data was not split by training.split_dataset, so "
        "train/test leakage (for example frames of one video in both) cannot be ruled out."
    )
    return None


def run_training(cfg: TrainConfig) -> dict:
    cfg.validate()
    set_seed(cfg.seed)
    data_dir, run_dir = Path(cfg.data_dir), Path(cfg.output_dir)

    if run_dir.exists() and any(run_dir.iterdir()) and not cfg.resume:
        raise TrainingError(f"{run_dir} is not empty. Choose a new --output-dir, or pass --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = _check_split_integrity(data_dir, cfg.require_manifest)
    train_samples, skipped_train = discover_samples(data_dir, "train")
    val_samples, skipped_val = discover_samples(data_dir, "validation")
    train_counts, val_counts = class_counts(train_samples), class_counts(val_samples)
    if min(train_counts) == 0 or min(val_counts) == 0:
        raise TrainingError("train and validation each need both classes (real and fake).")
    print(f"train: real {train_counts[0]}, fake {train_counts[1]}   "
          f"validation: real {val_counts[0]}, fake {val_counts[1]}")
    if skipped_train or skipped_val:
        print(f"WARNING: skipped {len(skipped_train) + len(skipped_val)} unreadable image(s)")

    device = resolve_device(cfg.device)
    augmenter = Augmenter() if cfg.augment else None
    common = dict(input_size=cfg.input_size, mean=cfg.mean, std=cfg.std)
    train_ds = ImageDataset(train_samples, augmenter=augmenter, seed=cfg.seed, **common)
    val_ds = ImageDataset(val_samples, **common)

    sampler = None
    if cfg.balance == "sampler":
        sampler = WeightedRandomSampler(
            sample_weights(train_samples, train_counts),
            num_samples=len(train_samples),
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        drop_last=len(train_ds) > cfg.batch_size,  # a final batch of 1 breaks BatchNorm
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg.architecture, cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    weight = (
        torch.tensor(class_weights(train_counts), dtype=torch.float32, device=device)
        if cfg.balance == "loss"
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)

    start_epoch, wait, history = 0, 0, []
    best = {"auc": -1.0, "loss": float("inf"), "epoch": -1}
    last_path = run_dir / "last.pt"
    if cfg.resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, wait, history = state["epoch"] + 1, state["best"], state["wait"], state["history"]
        print(f"Resuming after epoch {start_epoch}")
    else:
        (run_dir / "config.json").write_text(cfg.to_json(), encoding="utf-8")
        _write_json(run_dir / "run_info.json", {
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_counts": {"real": train_counts[0], "fake": train_counts[1]},
            "validation_counts": {"real": val_counts[0], "fake": val_counts[1]},
            "manifest_sha256": manifest_sha,
            "device": str(device),
            "torch": torch.__version__,
        })

    for epoch in range(start_epoch, cfg.epochs):
        started = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        running, seen = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(targets)
            seen += len(targets)
        scheduler.step()

        labels, probabilities, val_loss = predict_probabilities(model, val_loader, device)
        val = summarize_binary(labels, probabilities, band=tuple(cfg.uncertain_band))
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": float(val_loss),
            "val_accuracy": val["accuracy"],
            "val_f1": val["f1"],
            "val_roc_auc": val["roc_auc"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        auc = -1.0 if math.isnan(val["roc_auc"]) else val["roc_auc"]
        improved = auc > best["auc"] + 1e-9 or (abs(auc - best["auc"]) <= 1e-9 and val_loss < best["loss"])
        if improved:
            best = {"auc": float(auc), "loss": float(val_loss), "epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best.pt")
            wait = 0
        else:
            wait += 1

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
            "wait": wait, "history": history,
        }, last_path)
        _write_json(run_dir / "history.json", history)
        print(f"epoch {epoch + 1}/{cfg.epochs}  train_loss {_fmt(record['train_loss'])}  "
              f"val_loss {_fmt(record['val_loss'])}  val_acc {_fmt(record['val_accuracy'])}  "
              f"val_auc {_fmt(record['val_roc_auc'])}  {time.time() - started:.0f}s"
              f"{'  *best*' if improved else ''}")
        if wait >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} epochs.")
            break

    if not (run_dir / "best.pt").is_file():
        raise TrainingError("Training produced no checkpoint.")
    return {"run_dir": str(run_dir), "best_epoch": best["epoch"] + 1, "best_val_auc": best["auc"],
            "epochs_run": len(history)}


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(build_parser().parse_args(argv))
    try:
        summary = run_training(cfg)
    except (TrainingError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Done. Best validation ROC-AUC {_fmt(summary['best_val_auc'])} at epoch {summary['best_epoch']}. "
          f"Next: python -m training.evaluate --run {summary['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())