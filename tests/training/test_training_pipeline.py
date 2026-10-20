"""End-to-end checks on a toy dataset with a tiny network: they verify the plumbing
(splits, training, evaluation, export, backend loading), not detection quality."""

import csv
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from app.ml.model_meta import ModelMeta  # noqa: E402
from app.ml.preprocessing import load_image, resize_and_normalize  # noqa: E402
from training.augmentations import Augmenter  # noqa: E402
# from training.config import IMAGENET_PLACEHOLDER if False else TrainConfig  # noqa: E402,F401
from training.config import TrainConfig  # noqa: E402
from training.dataset import (  # noqa: E402
    ImageDataset,
    class_counts,
    class_weights,
    discover_samples,
)
from training.evaluate import evaluate_run, flatten_evaluation  # noqa: E402
from training.export_checkpoint import ExportError, export_checkpoint  # noqa: E402
from training.train import TrainingError, run_training  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def config(split_data, output_dir, **overrides):
    values = dict(
        data_dir=str(split_data), output_dir=str(output_dir), architecture="resnet18",
        pretrained=False, input_size=64, epochs=2, batch_size=16, num_workers=0, seed=0,
        device="cpu", patience=5,
    )
    values.update(overrides)
    return TrainConfig(**values)


@pytest.fixture(scope="module")
def trained_run(split_data, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("runs") / "run"
    run_training(config(split_data, run_dir))
    return run_dir


@pytest.fixture(scope="module")
def evaluated_run(trained_run, split_data):
    evaluate_run(trained_run, data_dir=split_data, batch_size=16, device="cpu")
    return trained_run


# ---- data ---------------------------------------------------------------------------------


def test_discovery_and_class_weights(split_data):
    samples, skipped = discover_samples(split_data, "train")
    counts = class_counts(samples)
    assert skipped == [] and min(counts) > 0
    assert class_weights([30, 10]) == pytest.approx([40 / 60, 2.0])
    with pytest.raises(ValueError):
        class_weights([5, 0])


def test_training_preprocessing_matches_the_servers(split_data):
    samples, _ = discover_samples(split_data, "train")
    dataset = ImageDataset(samples, input_size=64, mean=MEAN, std=STD)
    tensor, label = dataset[0]
    expected = resize_and_normalize(
        load_image(samples[0].path).rgb, 64, np.array(MEAN, np.float32), np.array(STD, np.float32)
    )
    assert tuple(tensor.shape) == (3, 64, 64)
    assert label == samples[0].label
    np.testing.assert_allclose(tensor.numpy(), expected)


def test_augmentation_is_reproducible_within_an_epoch(split_data):
    samples, _ = discover_samples(split_data, "train")
    dataset = ImageDataset(samples, input_size=64, mean=MEAN, std=STD, augmenter=Augmenter(), seed=7)
    dataset.set_epoch(3)
    assert np.array_equal(dataset[2][0].numpy(), dataset[2][0].numpy())


# ---- training -----------------------------------------------------------------------------


def test_training_refuses_a_leaking_split(split_data, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(split_data, copy)
    manifest = copy / "split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    next(r for r in rows if r["split"] == "train")["split"] = "test"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(TrainingError) as caught:
        run_training(config(copy, tmp_path / "run"))
    assert "more than one split" in str(caught.value)


def test_training_writes_checkpoints_and_history(trained_run):
    for name in ("best.pt", "last.pt", "history.json", "config.json", "run_info.json"):
        assert (trained_run / name).is_file()
    history = json.loads((trained_run / "history.json").read_text())
    assert len(history) == 2
    assert all(0.0 <= row["val_accuracy"] <= 1.0 for row in history)


def test_a_non_empty_output_directory_is_refused(trained_run, split_data):
    with pytest.raises(TrainingError):
        run_training(config(split_data, trained_run))


def test_training_can_be_resumed(split_data, tmp_path):
    run_dir = tmp_path / "run"
    run_training(config(split_data, run_dir, epochs=1))
    run_training(config(split_data, run_dir, epochs=2, resume=True))
    assert len(json.loads((run_dir / "history.json").read_text())) == 2


def test_the_sampler_balance_mode_runs(split_data, tmp_path):
    summary = run_training(config(split_data, tmp_path / "run", epochs=1, balance="sampler"))
    assert summary["epochs_run"] == 1


# ---- evaluation and export ----------------------------------------------------------------


def test_evaluation_report_is_complete_and_valid_json(evaluated_run):
    text = (evaluated_run / "evaluation_test.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    report = json.loads(text)
    assert report["split"] == "test"
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "ece",
            "brier", "uncertain_fraction"} <= set(report["metrics"])
    assert {"jpeg_q70", "downscale_50", "blur_1_5"} <= set(report["robustness"])


def test_flattened_evaluation_contains_only_numbers_and_text(evaluated_run):
    report = json.loads((evaluated_run / "evaluation_test.json").read_text())
    flat = flatten_evaluation(report, dataset_note="toy", trained_with="resnet18")
    assert all(isinstance(value, (float, str)) for value in flat.values())
    assert flat["test_images"] == report["num_images"]
    assert "robust_auc_jpeg_q70" in flat and flat["dataset"] == "toy"


def test_export_refuses_a_small_test_set(evaluated_run, tmp_path):
    with pytest.raises(ExportError) as caught:
        export_checkpoint(evaluated_run, tmp_path / "out", name="m", version="1", dataset_note="toy")
    assert "unreliable" in str(caught.value)


def test_export_requires_a_test_evaluation(evaluated_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(evaluated_run, copy)
    (copy / "evaluation_test.json").unlink()
    with pytest.raises(ExportError):
        export_checkpoint(copy, tmp_path / "out", name="m", version="1", dataset_note="toy",
                          force_small_test_set=True)


def test_exported_model_is_trained_tier_and_loads_with_a_heatmap(evaluated_run, tmp_path):
    out = export_checkpoint(
        evaluated_run, tmp_path / "out", name="toy-detector", version="0.0.1",
        dataset_note="TOY synthetic data", force_small_test_set=True,
    )
    meta = ModelMeta.model_validate(json.loads((out / "model_meta.json").read_text()))
    assert meta.tier == "TRAINED"
    assert meta.calibrated is False
    assert meta.gradcam_layer == "layer4"
    assert meta.evaluation and "roc_auc" in meta.evaluation
    assert "unreliable" in meta.evaluation["warning"]

    from app.config import Settings
    from app.ml.preprocessing import PreparedImage
    from app.ml.torch_detector import load_torch_detector

    detector = load_torch_detector(
        Settings(_env_file=None, detector_backend="torch", model_path=out, device="cpu")
    )
    assert detector.info.tier.value == "TRAINED"
    assert detector.capabilities.supports_heatmap is True
    rgb = np.random.default_rng(0).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    rgb.flags.writeable = False
    image = PreparedImage(rgb=rgb, format="png", mode="RGB", has_alpha=False, has_exif=False)
    assert 0.0 <= detector.predict_image(image).ai_probability <= 1.0