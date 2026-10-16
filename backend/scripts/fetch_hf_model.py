"""Download a Hugging Face image-classification model for TruthLens AI.

Run from backend/ with the optional dependencies installed (requirements-torch.txt):

    python scripts/fetch_hf_model.py --repo OWNER/MODEL --revision COMMIT_HASH \
        --license "licence from the model card" --out weights/my-model

First run: downloads the model and prints its labels. Re-run with
--synthetic-label LABEL (repeat for several) naming the label(s) that mean
"AI-generated", and the script writes model_meta.json. Read the model card first:
licence, training data and what the authors actually measured.
"""

import argparse
import json
import sys
from pathlib import Path

PATTERNS = ["*.json", "*.safetensors", "*.txt"]  # never pickled .bin/.pt files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--repo", required=True, help="Hugging Face repo id, OWNER/MODEL")
    parser.add_argument("--out", required=True, type=Path, help="Target directory")
    parser.add_argument("--license", required=True, help="Licence stated on the model card")
    parser.add_argument("--revision", default="main", help="Commit hash to pin (recommended)")
    parser.add_argument("--name", help="Display name (default: repo name)")
    parser.add_argument("--synthetic-label", action="append", default=[], help="Label meaning AI-generated")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Run: pip install -r backend/requirements-torch.txt", file=sys.stderr)
        return 1

    if args.revision == "main":
        print("Warning: revision 'main' is not pinned; the model can change under you. Prefer a commit hash.")

    out = args.out.resolve()
    snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=out, allow_patterns=PATTERNS)

    config_path = out / "config.json"
    if not config_path.is_file():
        print("No config.json downloaded: this does not look like a transformers model.", file=sys.stderr)
        return 1
    if not list(out.glob("*.safetensors")):
        print(
            "This repository has no .safetensors weights. Pickled weights are refused for safety; "
            "choose another model.",
            file=sys.stderr,
        )
        return 1

    id2label = json.loads(config_path.read_text(encoding="utf-8")).get("id2label", {})
    labels = [label for _, label in sorted(id2label.items(), key=lambda item: int(item[0]))]
    print(f"Model labels: {labels}")

    chosen = [label.strip() for label in args.synthetic_label]
    if not chosen:
        print("Re-run with --synthetic-label <label> naming the label(s) that mean AI-generated.")
        return 2
    known = {label.casefold() for label in labels}
    unknown = [label for label in chosen if label.casefold() not in known]
    if unknown:
        print(f"Unknown label(s) {unknown}. Choose from {labels}.", file=sys.stderr)
        return 2

    name = args.name or args.repo.split("/")[-1]
    meta = {
        "name": name,
        "version": args.revision[:12] if args.revision != "main" else "unpinned",
        "description": (
            f"Third-party pretrained image classifier '{args.repo}'. Not validated on this "
            "project's data; accuracy on new generators and on compressed or resized images is unknown."
        ),
        "tier": "PRETRAINED_ML",
        "kind": "huggingface",
        "synthetic_labels": chosen,
        "source": f"https://huggingface.co/{args.repo}@{args.revision}",
        "license": args.license,
        "calibrated": False,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {out / 'model_meta.json'}\nAdd this to your .env and restart the backend:\n")
    print("DETECTOR_BACKEND=torch")
    print(f"MODEL_PATH={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())