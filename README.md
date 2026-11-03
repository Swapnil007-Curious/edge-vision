# edge-vision

Landmark preprocessing and signer-independent splits for the Kaggle
[Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
dataset (`asl-signs`). This repository currently covers data preparation only: there is no model
or training code yet.

## Setup

Python 3.11, from the repository root (Windows PowerShell shown):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

## Preprocessing

```powershell
python scripts/preprocess.py --help
python scripts/preprocess.py --data-root <asl-signs dir> --output-dir <output dir>
```

Settings live in `experiments/preprocess_default.yaml`, whose paths point at Kaggle.
`--data-root`, `--output-dir` and `--limit` override the YAML. `--no-cache` skips the per-sequence
cache in `<output dir>/cache/`, which otherwise makes reruns resumable.

The run writes `features.npz`, `manifest.csv`, `preprocess_report.json` and
`splits_<mode>.json` to the output directory.

## Tests

```powershell
ruff check .
ruff format --check .
mypy --strict src
pytest -q
```

Tests use synthetic fixtures only; the real dataset is not needed.

## Results

Results are generated later; none are reported yet.
