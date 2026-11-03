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

`requirements.txt` holds the package dependencies (preprocessing, plus py-cpuinfo for benchmark
records), `requirements-train.txt` adds torch, onnx and onnxruntime, and `requirements-dev.txt`
adds the test and lint tools.

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
  split, for the checkpoints and (with `--quant-root`) their quantised variants, and it logs
  every invocation to `results/final/test_access_log.json`.
- **Seeds.** Each signer-independent configuration is trained with seeds 0, 1 and 2, and results
  are reported as the mean and standard deviation over seeds. Runs are seeded and use
  deterministic algorithms where available; GPU runs may still differ bitwise between repeats.

## Quantisation

Post-training quantisation with ONNX Runtime, for every completed signer-independent run:

```powershell
python scripts/quantize.py --runs-root results/runs --processed-dir data/processed
python scripts/final_eval.py --processed-dir data/processed --quant-root results/quant
```

`quantize.py` writes four variants per run to `results/quant/<model>_signer_independent/seed_<s>/`:

- `fp32.onnx`: the checkpoint exported at opset 17, velocity and standardisation included.
- `fp16.onnx`: float16 weights and compute with float32 input and output. If any val output is
  non-finite or val top-1 agreement with fp32 is below 99%, it is rebuilt with the feature block
  (velocity and standardisation) kept in float32, and the report says so.
- `int8_dynamic.onnx`: dynamic quantisation, int8 weights per channel.
- `int8_static.onnx`: static QDQ quantisation after `quant_pre_process`, uint8 activations and
  int8 per-channel weights, MinMax calibration on 1024 train sequences sampled with seed 0 and
  fed in batches of 64. Calibration reads only the train split.

Each run's `quant_report.json` records, per variant, the file size, op counts, which MatMul, Gemm,
Conv and GRU nodes stayed float, the checker and onnxruntime load results, val top-1, val top-1
agreement with fp32 and the largest logit difference from fp32 on 512 val sequences. A variant
that fails is recorded there and the other variants still run. `results/tables/quant_ops_summary.csv`
summarises the reports. Existing variant files are reused unless `--overwrite` is passed; a file
built from a different `best.pt` is reported as stale.

`final_eval.py --quant-root` evaluates every variant on test, first checking that each fp32
variant's test top-1 matches its checkpoint's within 0.1 percentage points. It writes
`results/final/quant/<model>/seed_<s>/<variant>_test_metrics.json` and
`results/tables/test_quant_summary.csv`, where deltas against fp32 are paired by seed. Without
`--quant-root` the script behaves as before. The `.onnx` files are not tracked by git; the reports
and tables are.

## Benchmarking

```powershell
python scripts/benchmark.py
```

The benchmark measures CPU latency for one sequence per call. Defaults: models `gru`, `cnn` and
`transformer` at seed 0, variants `torch_fp32`, `onnx_fp32`, `fp16`, `int8_dynamic` and
`int8_static`, and 1, 2 and 4 intra-op threads. `--batch N` switches to a throughput mode whose
output files carry a `_batch<N>` suffix.

Measurement protocol:

- **Inputs.** Val sequences in split order, cycled. The test split is never read.
- **Sessions.** Each configuration gets a fresh session: onnxruntime with `ORT_SEQUENTIAL`, one
  inter-op thread and `ORT_ENABLE_ALL`, or eager torch under `torch.inference_mode` with
  `torch.set_num_threads`. Session creation time and the approximate growth in process RSS are
  recorded.
- **Timing.** 100 warm-up calls, then 2000 calls each timed on its own with `time.perf_counter_ns`,
  with Python's garbage collector paused. The run reports mean, p50, p95, p99, min and max in ms,
  and throughput in sequences per second.
- **Rounds.** The full configuration list runs 3 times, in a new seeded random order each round,
  with a 3 s pause between configurations. The summary reports medians across rounds, plus
  `p50_spread_ms`, the largest minus the smallest p50. With the defaults, the pauses alone take
  about 7 minutes.
- **Machine state.** Run on AC power, with the Windows power mode set to Best performance and
  other applications closed. `results/bench/bench_env.json` records the CPU flags detected at
  runtime (AVX2, AVX-512, VNNI), the clock at the start and end, the power source, the active
  power scheme and power mode, and the library versions. A run on battery prints a warning and
  records it there.

Outputs: `results/bench/bench_raw.csv` (one row per configuration and round),
`results/bench/bench_env.json` and `results/tables/bench_summary.csv`. The summary's
`p95_share_of_33ms_frame` column is p95 latency as a share of one 30 fps frame.

## Tests

```powershell
ruff check .
ruff format --check .
mypy --strict src
pytest -q
```

Tests use synthetic fixtures only; the real dataset is not needed.

## Results

Result tables are generated into `results/tables/` by `scripts/aggregate_val.py`,
`scripts/final_eval.py`, `scripts/quantize.py` and `scripts/benchmark.py`. None are reported here
yet.
