"""Create a TOY dataset so the whole pipeline can be tried without real data.

"real" images are smooth colour gradients; "fake" images are the same gradients with a fine
checkerboard added. That is trivially separable and has nothing to do with real AI-image
detection: a model trained on it is NOT a detector. Groups of 3 images share the same
gradient, so the split script has real near-duplicates to keep together.

    python -m training.make_sample_dataset --out raw_sample
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _image(group_seed: int, noise_rng: np.random.Generator, label: str, size: int) -> np.ndarray:
    colors = np.random.default_rng(group_seed)
    start, end = colors.uniform(40, 200, 3), colors.uniform(40, 200, 3)
    ramp = np.linspace(0.0, 1.0, size)[None, :, None]
    image = np.broadcast_to(start * (1 - ramp) + end * ramp, (size, size, 3)).copy()
    if label == "fake":
        checker = ((np.indices((size, size)).sum(axis=0) % 2) * 2 - 1)[..., None] * 18.0
        image += checker
    image += noise_rng.normal(0.0, 3.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_dataset(
    out, *, groups_per_class: int = 30, images_per_group: int = 3, size: int = 64, seed: int = 0
) -> int:
    rng = np.random.default_rng(seed)
    out = Path(out)
    total = 0
    for label, prefix in (("real", "r"), ("fake", "f")):
        folder = out / label
        folder.mkdir(parents=True, exist_ok=True)
        for group in range(groups_per_class):
            group_seed = int(rng.integers(0, 2**31 - 1))
            for k in range(images_per_group):
                Image.fromarray(_image(group_seed, rng, label, size)).save(
                    folder / f"{prefix}{group:04d}__{k}.png"
                )
                total += 1
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a toy dataset (not a real detector dataset).")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--groups-per-class", type=int, default=30)
    parser.add_argument("--images-per-group", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    total = make_dataset(
        args.out, groups_per_class=args.groups_per_class,
        images_per_group=args.images_per_group, size=args.size, seed=args.seed,
    )
    print(f"Wrote {total} TOY images to {args.out}. A model trained on them is not a real detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())