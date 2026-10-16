import json

import pytest

from app.ml import model_meta
from app.ml.model_meta import (
    ModelMeta,
    load_model_meta,
    resolve_model_paths,
    safe_weights_path,
)
from app.utils.errors import ModelError

HF = {
    "name": "m",
    "version": "1",
    "description": "d",
    "tier": "PRETRAINED_ML",
    "kind": "huggingface",
    "synthetic_labels": ["ai"],
}
TV = {**HF, "kind": "torchvision", "architecture": "resnet18", "class_names": ["real", "fake"]}


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_metas_parse():
    assert ModelMeta.model_validate(HF).calibrated is False
    assert ModelMeta.model_validate(TV).input_size == 224


def test_unknown_fields_are_rejected(tmp_path):
    with pytest.raises(ModelError) as caught:
        load_model_meta(write(tmp_path / "m.json", {**HF, "tyre": "PRETRAINED_ML"}))
    assert "tyre" in str(caught.value)


def test_trained_tier_requires_evaluation(tmp_path):
    with pytest.raises(ModelError):
        load_model_meta(write(tmp_path / "m.json", {**HF, "tier": "TRAINED"}))
    ok = {**HF, "tier": "TRAINED", "evaluation": {"roc_auc": 0.93}}
    assert load_model_meta(write(tmp_path / "ok.json", ok)).tier == "TRAINED"


def test_calibrated_only_for_trained(tmp_path):
    with pytest.raises(ModelError):
        load_model_meta(write(tmp_path / "m.json", {**HF, "calibrated": True}))


def test_torchvision_needs_architecture_and_classes(tmp_path):
    for broken in ({**TV, "architecture": None}, {**TV, "class_names": ["only-one"]}):
        with pytest.raises(ModelError):
            load_model_meta(write(tmp_path / "m.json", broken))


def test_empty_synthetic_labels_rejected(tmp_path):
    with pytest.raises(ModelError):
        load_model_meta(write(tmp_path / "m.json", {**HF, "synthetic_labels": []}))


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelError):
        load_model_meta(path)


def test_model_path_is_required():
    with pytest.raises(ModelError):
        resolve_model_paths(None, None)


def test_paths_must_exist(tmp_path):
    with pytest.raises(ModelError):
        resolve_model_paths(tmp_path / "missing", None)
    (tmp_path / "dir").mkdir()
    with pytest.raises(ModelError):  # directory exists, metadata does not
        resolve_model_paths(tmp_path / "dir", None)


def test_default_meta_location_and_relative_paths_use_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(model_meta, "REPO_ROOT", tmp_path)
    directory = tmp_path / "weights" / "m"
    directory.mkdir(parents=True)
    write(directory / "model_meta.json", HF)

    resolved_dir, resolved_meta = resolve_model_paths(model_meta.Path("weights/m"), None)
    assert resolved_dir == directory.resolve()
    assert resolved_meta == (directory / "model_meta.json").resolve()


def test_weights_path_cannot_escape_the_model_directory(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    (tmp_path / "evil.pt").write_bytes(b"x")
    (directory / "model.pt").write_bytes(b"x")
    (directory / "model.bin").write_bytes(b"x")

    assert safe_weights_path(directory, "model.pt").name == "model.pt"
    with pytest.raises(ModelError):
        safe_weights_path(directory, "../evil.pt")
    with pytest.raises(ModelError):
        safe_weights_path(directory, str(tmp_path / "evil.pt"))
    with pytest.raises(ModelError):  # pickle-style suffixes outside the allowlist
        safe_weights_path(directory, "model.bin")
    with pytest.raises(ModelError):
        safe_weights_path(directory, "missing.pt")