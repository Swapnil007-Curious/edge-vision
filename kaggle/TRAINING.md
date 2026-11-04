# Training on Kaggle

This notebook trains the full grid, aggregates validation results, evaluates once on the test
split and bundles everything into `/kaggle/working/results_bundle.zip`. The v2 ablation grid has
its own notebook; see [v2](#v2).

## Before you start

- Push the current state of this repository to GitHub; the notebook clones it.
- Run the preprocessing notebook (`scripts/preprocess.py`) and commit it with **Save & Run All**,
  so its `/kaggle/working/processed` output is saved.

## Notebook settings

- **Accelerator: GPU** (T4 or P100), under Settings > Accelerator. Training without a GPU falls
  back to the CPU and is far too slow for the full grid.
- **Internet: On**, for `git clone` and for pip to fetch the build backend.
- **Add the preprocessing output as an input**: in the editor's right-hand panel choose
  Input > Add Input > Your Work > Notebooks, pick the preprocessing notebook and click Add. Its
  saved output then appears under `/kaggle/input/`.
- Commit with **Save Version > Save & Run All (Commit)**. Only a committed version keeps its
  outputs; `results_bundle.zip` is then in that version's Output tab.

## Cells

One command per cell, in order.

```
!git clone https://github.com/Swapnil007-Curious/edge-vision.git /kaggle/working/edge-vision
```

```
%cd /kaggle/working/edge-vision
```

```
!pip install -q --no-deps -e .
```

```
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('kernel check', (torch.ones(4, device='cuda') * 2).sum().item())"
```

```
!python -c "import onnx, onnxruntime; print('onnx', onnx.__version__, 'onnxruntime', onnxruntime.__version__)" || pip install -q onnx onnxruntime
```

```
!ls /kaggle/input
```

```
!find /kaggle/input -name features.npz
```

Set `PROCESSED` to the directory that holds the `features.npz` printed above:

```
PROCESSED = "/kaggle/input/edge-vision-preprocess/processed"
```

```
!python scripts/run_grid.py --processed-dir {PROCESSED} --output-root /kaggle/working/edge-vision/results/runs
```

```
!python scripts/aggregate_val.py
```

```
!python scripts/final_eval.py
```

```
!python scripts/pack_results.py --output /kaggle/working/results_bundle.zip
```

## Notes

- The grid is 9 signer-independent runs (gru, cnn and transformer at seeds 0, 1 and 2) and 3
  signer-dependent runs (each model at seed 0). Each run prints one line per epoch.
- `run_grid.py` skips runs whose `summary.json` exists. Within one session, rerunning its cell
  resumes where it stopped. A new version starts with an empty `/kaggle/working`; to carry
  finished runs over, add the earlier version's output as an input and, before the `run_grid.py`
  cell, run `!cp -r /kaggle/input/<earlier-version>/edge-vision/results /kaggle/working/edge-vision/`.
- If the kernel check fails with "no kernel image is available for execution on the device",
  the preinstalled torch does not support that GPU; switch the accelerator to T4.
- `final_eval.py` is the only step that reads the test split. It appends every invocation to
  `results/final/test_access_log.json`, so run it once, after model selection on validation.
- The processed input is read-only and is never written; split files a run needs are written to
  that run's directory.

## v2

The v2 notebook preprocesses with the v2 landmark selection (pose 0-16) in the same session,
trains the ablation grid (experiments b, c and d at seeds 0, 1 and 2), then aggregates, evaluates
once on test and bundles everything into `/kaggle/working/results_bundle_v2.zip`.

### Notebook settings

As above (**GPU**, **Internet: On**, commit with **Save & Run All**), but add the competition
data instead of the preprocessing output: Input > Add Input > Competitions > Google - Isolated
Sign Language Recognition.

### Cells

One command per cell, in order.

```
!git clone https://github.com/Swapnil007-Curious/edge-vision.git /kaggle/working/edge-vision
```

```
%cd /kaggle/working/edge-vision
```

```
!pip install -q --no-deps -e .
```

```
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('kernel check', (torch.ones(4, device='cuda') * 2).sum().item())"
```

```
!python -c "import onnx, onnxruntime; print('onnx', onnx.__version__, 'onnxruntime', onnxruntime.__version__)" || pip install -q onnx onnxruntime
```

```
!find /kaggle/input -maxdepth 3 -name train.csv
```

Set `DATA_ROOT` to the directory that holds the `train.csv` printed above:

```
DATA_ROOT = "/kaggle/input/competitions/asl-signs"
```

```
!python scripts/preprocess.py --config experiments/preprocess_v2.yaml --data-root {DATA_ROOT} --output-dir /kaggle/working/processed_v2
```

Check that the v2 split holds the same participants as the v1 split committed in the repository:

```
!python -c "import json; v2 = json.load(open('/kaggle/working/processed_v2/splits_signer_independent.json'))['participants']; v1 = json.load(open('results/runs/cnn_signer_independent/seed_0/splits_signer_independent.json'))['participants']; print('same participants as v1:', v2 == v1); assert v2 == v1"
```

```
!python scripts/run_grid.py --grid v2 --processed-dir /kaggle/working/processed_v2 --output-root /kaggle/working/edge-vision/results/runs --max-hours 10
```

```
!python scripts/aggregate_val.py
```

```
!python scripts/final_eval.py --processed-dir /kaggle/working/processed_v2
```

```
!python scripts/pack_results.py --output /kaggle/working/results_bundle_v2.zip
```

### Notes

- The grid order is b, c, d at seed 0, then seed 1, then seed 2, so an interrupted grid still has
  every experiment at seed 0. `--max-hours 10` stops starting new runs after 10 hours of grid time,
  which leaves room for preprocessing and evaluation in a 12-hour session; the skipped runs are
  printed. To finish them, carry the results over to a new version as described above and rerun
  the `run_grid.py` cell; completed runs are skipped.
- The clone carries the committed v1 run summaries, but not their checkpoints (`*.pt` is not in
  git). `aggregate_val.py` includes them in `val_summary.csv`, so the v1 CNN
  (`cnn_signer_independent`) sits beside b, c and d for comparison. `final_eval.py` leaves runs
  without `best.pt` out, so the bundle's test tables cover the v2 experiments, and it appends one
  entry to the committed `test_access_log.json`.
- Seed ensembles: `aggregate_val.py` writes `results/tables/val_ensemble.csv` and `final_eval.py`
  writes `results/tables/test_ensemble.csv`, one row per experiment with at least two seeds.
- The v2 runs are not quantised: `quantize.py` lists and skips runs with v2 features.
