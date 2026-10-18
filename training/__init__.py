"""Training pipeline for TruthLens AI. Run with `python -m training.<module>` from the repo root.

The backend directory is put on sys.path so training reuses the server's own image decoding
and preprocessing (app.ml.preprocessing): what the model sees in training is exactly what
it sees when serving.
"""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))