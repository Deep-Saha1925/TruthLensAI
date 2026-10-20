"""Export a trained run as a model folder the backend can load.

    python -m training.export_checkpoint --run runs/exp1 --out backend/weights/my-detector \
        --name my-detector --version 1.0.0 --dataset-note "what the data was"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from training.config import CLASS_NAMES, GRADCAM_LAYERS, TrainConfig
from training.evaluate import flatten_evaluation
from training.modeling import build_model


class ExportError(Exception):
    pass


def export_checkpoint(
    run_dir,
    out_dir,
    *,
    name: str,
    version: str,
    dataset_note: str,
    license_text: str = "Not specified",
    min_test_images: int = 200,
    force_small_test_set: bool = False,
) -> Path:
    run_dir, out = Path(run_dir), Path(out_dir)
    cfg = TrainConfig.from_json((run_dir / "config.json").read_text(encoding="utf-8"))

    report_path = run_dir / "evaluation_test.json"
    if not report_path.is_file():
        raise ExportError(
            "evaluation_test.json is missing. Run `python -m training.evaluate --run ...` on the "
            "test split first: a model without measured performance cannot be exported as TRAINED."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "test":
        raise ExportError("The evaluation must come from the test split.")

    test_images = int(report["num_images"])
    warning = None
    if test_images < min_test_images:
        if not force_small_test_set:
            raise ExportError(
                f"The test split has only {test_images} images (minimum {min_test_images}); the "
                "metrics would be unreliable. Use more data, or pass --force-small-test-set for a "
                "demo or toy export."
            )
        warning = f"Evaluated on only {test_images} test images: these metrics are unreliable."

    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    build_model(cfg.architecture, pretrained=False).load_state_dict(state, strict=True)  # must match

    initialization = "ImageNet-pretrained initialization" if cfg.pretrained else "random initialization"
    trained_with = f"{cfg.architecture}, {initialization}, {report.get('epochs_trained') or '?'} epochs"
    evaluation = flatten_evaluation(report, dataset_note=dataset_note, trained_with=trained_with)
    if warning:
        evaluation["warning"] = warning

    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "model.pt")
    shutil.copy2(report_path, out / "evaluation_report.json")
    meta = {
        "name": name,
        "version": version,
        "description": (
            f"{cfg.architecture} trained with this project's pipeline ({initialization}) on: "
            f"{dataset_note}. Metrics come from a held-out test split of that data and do not "
            "transfer automatically to other generators, cameras or platforms."
        ),
        "tier": "TRAINED",
        "kind": "torchvision",
        "architecture": cfg.architecture,
        "class_names": CLASS_NAMES,
        "synthetic_labels": ["fake"],
        "weights_file": "model.pt",
        "input_size": cfg.input_size,
        "mean": cfg.mean,
        "std": cfg.std,
        "gradcam_layer": GRADCAM_LAYERS[cfg.architecture],
        "source": f"Trained with this project's pipeline (run '{run_dir.name}')",
        "license": license_text,
        "calibrated": False,  # calibration error is measured, but no calibration is applied
        "evaluation": evaluation,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Final check: load it exactly like the server will.
    from app.config import Settings
    from app.ml.torch_detector import load_torch_detector

    try:
        load_torch_detector(
            Settings(_env_file=None, detector_backend="torch", model_path=out.resolve(), device="cpu")
        )
    except Exception as exc:
        raise ExportError(f"The exported model failed the backend's load check: {exc}") from exc
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a trained run for the backend.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dataset-note", required=True, help="What the data was (shown to users)")
    parser.add_argument("--license", default="Not specified", dest="license_text")
    parser.add_argument("--min-test-images", type=int, default=200)
    parser.add_argument("--force-small-test-set", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = export_checkpoint(
            args.run, args.out, name=args.name, version=args.version, dataset_note=args.dataset_note,
            license_text=args.license_text, min_test_images=args.min_test_images,
            force_small_test_set=args.force_small_test_set,
        )
    except (ExportError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Exported to {out.resolve()}\n\nAdd to your .env and restart the backend:\n")
    print("DETECTOR_BACKEND=torch")
    print(f"MODEL_PATH={out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())