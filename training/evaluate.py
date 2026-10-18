"""Evaluate a trained run on a split (use the TEST split, once, after you stop tuning).

    python -m training.evaluate --run runs/exp1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from training.augmentations import downscale, gaussian_blur, jpeg_compress
from training.config import CLASS_NAMES, TrainConfig
from training.dataset import ImageDataset, class_counts, discover_samples
from training.metrics import json_safe, roc_curve, summarize_binary
from training.modeling import build_model, load_weights, predict_probabilities, resolve_device
from training.split_dataset import MANIFEST_NAME, file_sha256

# Picklable (functools.partial) so they also work with DataLoader worker processes on Windows.
CONDITIONS = {
    "jpeg_q90": partial(jpeg_compress, quality=90),
    "jpeg_q70": partial(jpeg_compress, quality=70),
    "jpeg_q50": partial(jpeg_compress, quality=50),
    "downscale_50": partial(downscale, factor=0.5),
    "downscale_25": partial(downscale, factor=0.25),
    "blur_1_5": partial(gaussian_blur, sigma=1.5),
}


class EvaluationError(Exception):
    pass


def _score(model, samples, cfg, device, degrade, batch_size, num_workers):
    dataset = ImageDataset(
        samples, input_size=cfg.input_size, mean=cfg.mean, std=cfg.std, degrade=degrade
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return predict_probabilities(model, loader, device)


def save_plots(run_dir: Path, split: str, labels, probabilities, metrics) -> bool:
    """ROC curve and confusion matrix PNGs. Skipped (returns False) without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not math.isnan(metrics["roc_auc"]):
        fpr, tpr = roc_curve(labels, probabilities)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(fpr, tpr, label=f"ROC-AUC {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve ({split})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(run_dir / f"roc_{split}.png", dpi=150)
        plt.close(fig)

    matrix = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], [f"pred {n}" for n in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"true {n}" for n in CLASS_NAMES])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, matrix[row][col], ha="center", va="center")
    ax.set_title(f"Confusion matrix ({split})")
    fig.tight_layout()
    fig.savefig(run_dir / f"confusion_{split}.png", dpi=150)
    plt.close(fig)
    return True


def evaluate_run(
    run_dir,
    *,
    data_dir=None,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 0,
    robustness: bool = True,
    device: str = "auto",
) -> dict:
    run_dir = Path(run_dir)
    cfg = TrainConfig.from_json((run_dir / "config.json").read_text(encoding="utf-8"))
    data = Path(data_dir or cfg.data_dir)

    samples, skipped = discover_samples(data, split)
    counts = class_counts(samples)
    if min(counts) == 0:
        raise EvaluationError(f"The {split} split needs images of both classes (found {counts}).")

    dev = resolve_device(device)
    model = build_model(cfg.architecture, pretrained=False)
    load_weights(model, run_dir / "best.pt")
    model.to(dev)

    band = tuple(cfg.uncertain_band)
    labels, probabilities, _ = _score(model, samples, cfg, dev, None, batch_size, num_workers)
    metrics = summarize_binary(labels, probabilities, band=band)

    robust = {}
    if robustness:
        for name, degrade in CONDITIONS.items():
            l, p, _ = _score(model, samples, cfg, dev, degrade, batch_size, num_workers)
            summary = summarize_binary(l, p, band=band)
            robust[name] = {"roc_auc": summary["roc_auc"], "accuracy": summary["accuracy"]}

    manifest = data / MANIFEST_NAME
    history_path = run_dir / "history.json"
    report = {
        "split": split,
        "num_images": len(samples),
        "class_counts": dict(zip(CLASS_NAMES, counts)),
        "unreadable_skipped": len(skipped),
        "metrics": metrics,
        "robustness": robust,
        "manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": cfg.architecture,
        "pretrained_init": cfg.pretrained,
        "epochs_trained": len(json.loads(history_path.read_text())) if history_path.is_file() else None,
        "uncertain_band": list(band),
    }
    (run_dir / f"evaluation_{split}.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    save_plots(run_dir, split, labels, probabilities, metrics)
    return report


def flatten_evaluation(report: dict, *, dataset_note: str, trained_with: str) -> dict:
    """The flat {name: number | text} dict stored in model_meta.json and shown in the UI."""
    flat: dict[str, float | str] = {}

    def put(key: str, value) -> None:
        if isinstance(value, (int, float)) and math.isfinite(value):
            flat[key] = float(value)

    metrics = report["metrics"]
    put("test_images", report["num_images"])
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "ece", "brier",
                "uncertain_fraction", "accuracy_outside_uncertain_band"):
        put(key, metrics.get(key))
    for condition, values in report.get("robustness", {}).items():
        put(f"robust_auc_{condition}", values.get("roc_auc"))
        put(f"robust_acc_{condition}", values.get("accuracy"))

    low, high = report["uncertain_band"]
    flat["dataset"] = dataset_note
    flat["trained_with"] = trained_with
    flat["threshold"] = (
        f"Precision, recall and F1 use a 0.5 cut-off on the fake score; 'uncertain' means a score "
        f"between {low} and {high}."
    )
    if report.get("manifest_sha256"):
        flat["manifest_sha256"] = report["manifest_sha256"][:16]
    flat["evaluated_at"] = report["evaluated_at"]
    return flat


def _print_report(report: dict) -> None:
    m = report["metrics"]

    def f(value):
        return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"

    print(f"\n{report['split']} split: {report['num_images']} images {report['class_counts']}")
    print(f"  accuracy {f(m['accuracy'])}   precision {f(m['precision'])}   recall {f(m['recall'])}   F1 {f(m['f1'])}")
    print(f"  ROC-AUC {f(m['roc_auc'])}   ECE {f(m['ece'])}   Brier {f(m['brier'])}")
    print(f"  uncertain share {f(m['uncertain_fraction'])}   accuracy outside band {f(m['accuracy_outside_uncertain_band'])}")
    print(f"  confusion matrix (rows true real/fake, cols predicted): {m['confusion_matrix']}")
    if report["robustness"]:
        print("  robustness (ROC-AUC / accuracy):")
        for name, values in report["robustness"].items():
            print(f"    {name:<13} {f(values['roc_auc'])} / {f(values['accuracy'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a training run.")
    parser.add_argument("--run", required=True, type=Path, help="Run folder produced by training.train")
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to the run's data dir")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-robustness", dest="robustness", action="store_false")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        report = evaluate_run(
            args.run, data_dir=args.data_dir, split=args.split, batch_size=args.batch_size,
            num_workers=args.num_workers, robustness=args.robustness, device=args.device,
        )
    except (EvaluationError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    print(f"\nWrote {args.run / f'evaluation_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())