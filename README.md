# edge-vision

Landmark preprocessing, signer-independent splits and compact sequence classifiers for the
Kaggle [Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
dataset (`asl-signs`). Three models (bidirectional GRU, temporal CNN, small transformer) are
trained on preprocessed hand and pose landmarks and export to ONNX.

## Setup

Python 3.11, from the repository root (Windows PowerShell shown). torch is the CPU build from the
PyTorch index, so it is installed first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e .
```

`requirements.txt` holds the preprocessing dependencies, `requirements-train.txt` adds torch,
onnx and onnxruntime, and `requirements-dev.txt` adds the test and lint tools.

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

## Training

Each model has a config in `experiments/train_<model>.yaml` (`gru`, `cnn`, `transformer`). A short
local smoke run on CPU:

```powershell
python scripts/train.py --config experiments/train_gru.yaml --seed 0 --processed-dir <processed dir> --epochs 2 --device cpu
```

A run writes `config.yaml`, `metrics.csv`, `best.pt`, `label_map.json`, `summary.json` and the
split file it used to `results/runs/<model>_<split_mode>/seed_<seed>/`. It refuses to replace a
completed run unless `--overwrite` is passed. The processed directory is only read.

Full training runs on a Kaggle GPU; see [kaggle/TRAINING.md](kaggle/TRAINING.md) for the notebook
cells.

## Evaluation protocol

- **Signer-independent split.** Participants are assigned whole to train, validation or test (4
  test and 3 validation signers), so test signers are never seen in training. A signer-dependent
  stratified split is trained only as a comparison.
- **Model selection on validation only.** Checkpoints, early stopping and the choice of best model
  use validation top-1. `scripts/aggregate_val.py` summarises validation metrics from the runs'
  summaries and never reads the test split.
- **One test evaluation script.** `scripts/final_eval.py` is the only code that loads the test
  split, and it logs every invocation to `results/final/test_access_log.json`.
- **Seeds.** Each signer-independent configuration is trained with seeds 0, 1 and 2, and results
  are reported as the mean and standard deviation over seeds. Runs are seeded and use
  deterministic algorithms where available; GPU runs may still differ bitwise between repeats.

## Tests

```powershell
ruff check .
ruff format --check .
mypy --strict src
pytest -q
```

Tests use synthetic fixtures only; the real dataset is not needed.

## Results

Result tables are generated into `results/tables/` by `scripts/aggregate_val.py` and
`scripts/final_eval.py`. None are reported here yet.
