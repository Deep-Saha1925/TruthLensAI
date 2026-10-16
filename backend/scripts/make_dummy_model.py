"""Create a RANDOMLY INITIALISED torchvision model folder for testing the heatmap UI.

Its predictions are meaningless. Use it only to see the heatmap overlay working, then
switch back to a real model or the demo detector.

    python scripts/make_dummy_model.py weights/dummy-resnet18
"""

import json
import sys
from pathlib import Path

import torch
import torchvision.models as tv_models


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "weights/dummy-resnet18").resolve()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = tv_models.resnet18(weights=None, num_classes=2)
    torch.save(model.state_dict(), out / "model.pt")

    meta = {
        "name": "dummy-random-resnet18",
        "version": "0.0.0",
        "description": "RANDOMLY INITIALISED network for testing the heatmap UI. Its predictions are meaningless.",
        "tier": "PRETRAINED_ML",
        "kind": "torchvision",
        "architecture": "resnet18",
        "class_names": ["real", "fake"],
        "synthetic_labels": ["fake"],
        "input_size": 224,
        "gradcam_layer": "layer4",
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}\nDETECTOR_BACKEND=torch\nMODEL_PATH={out}")


if __name__ == "__main__":
    main()