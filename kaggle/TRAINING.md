# Training on Kaggle

This notebook trains the full grid, aggregates validation results, evaluates once on the test
split and bundles everything into `/kaggle/working/results_bundle.zip`.

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
