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
split file it used to `results/runs/<experiment>/seed_<seed>/`. A v1 config's experiment is
`<model>_<split_mode>` (e.g. `gru_signer_independent`); v2 configs name theirs. It refuses to
replace a completed run unless `--overwrite` is passed. The processed directory is only read.

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
- **Seed ensembles.** For every experiment with at least two completed seeds, `aggregate_val.py`
  also scores the mean of the seeds' logits on validation (`val_ensemble.csv`) and
  `final_eval.py` on test (`test_ensemble.csv`).
- **Experiments.** Every table starts with an `experiment` column. Runs are found from their
  `summary.json` fields (model, split mode, feature set, recipe); summaries written before these
  fields existed are v1 runs.

## v2 ablation: features, recipe and architecture

The v2 grid changes one thing at a time against the v1 CNN (`cnn_signer_independent`), with the
same signer-independent split:

| Experiment | Model | Features | Recipe |
| --- | --- | --- | --- |
| `b_cnn_v2feat_v1recipe` | cnn | v2 | v1 |
| `c_cnn_v2feat_v2recipe` | cnn | v2 | v2 |
| `d_hybrid_v2feat_v2recipe` | hybrid | v2 | v2 |

- **v2 landmarks** (`experiments/preprocess_v2.yaml`): pose 0-16 (nose, eyes, ears, mouth
  corners, shoulders, elbows, wrists), so L = 59. Mirroring swaps every selected left/right pose
  pair. `preprocess_report.json` records the selection (`landmarks`) and, for v1 and v2 alike, the
  face/shoulder `aspect_stats`.
- **v2 features** (`models/features_v2.py`, inside the ONNX graph): coordinates, velocity,
  acceleration, the 210 pairwise distances of the dominant hand and 24 distances from its wrist
  and fingertips to the nose, mouth centre and shoulders; 588 features, standardised with train
  statistics. Any value derived from a missing point is 0.0.
- **v2 recipe**: 150 epochs, patience 30, an exponential moving average of the weights (decay
  0.999; validation and `best.pt` use it), and extra augmentations: frame dropout, time masking,
  landmark masking and hand-shape rotation.
- **Hybrid** (`models/hybrid.py`): three stages of two depthwise-conv blocks and one transformer
  block, with late head dropout (off for the first half of the epochs).

```powershell
python scripts/preprocess.py --config experiments/preprocess_v2.yaml --data-root <asl-signs dir> --output-dir <processed_v2 dir>
python scripts/run_grid.py --grid v2 --processed-dir <processed_v2 dir> --max-hours 10
```

`--max-hours` stops starting new runs once that much wall time has passed; a rerun resumes.
`scripts/export_web.py` and quantisation handle v1-feature runs only.
`src/edgevision/export/ensemble.py` exports the seeds of one experiment as one ONNX file that
averages their logits.

## Quantisation

Post-training quantisation with ONNX Runtime, for every completed signer-independent run:

```powershell
python scripts/quantize.py --runs-root results/runs --processed-dir data/processed
python scripts/final_eval.py --processed-dir data/processed --quant-root results/quant
```

`quantize.py` writes five variants per run to `results/quant/<model>_signer_independent/seed_<s>/`:

- `fp32.onnx`: the checkpoint exported at opset 17, velocity and standardisation included.
- `fp16.onnx`: float16 weights and compute with float32 input and output. If any val output is
  non-finite or val top-1 agreement with fp32 is below 99%, it is rebuilt with the feature block
  (velocity and standardisation) kept in float32, and the report says so.
- `int8_dynamic.onnx`: dynamic quantisation, int8 weights per channel.
- `int8_static.onnx`: static QDQ quantisation after `quant_pre_process`, uint8 activations and
  int8 per-channel weights, MinMax calibration on 1024 train sequences sampled with seed 0 and
  fed in batches of 64. Calibration reads only the train split. The feature block's nodes are
  excluded from quantisation, so the `landmarks` input, velocity and standardisation stay
  float32 and quantisation starts at the input of the first projection layer. Quantising the
  raw input would round coordinates to an 8-bit grid before velocity is computed, and could turn
  small coordinates (common near the torso centre after shoulder normalisation) into exact
  zeros, which the feature block treats as missing points. The build fails if no feature-block
  node is found, or if any QuantizeLinear node still reads the input or a tensor inside the block.
- `int8_static_full.onnx`: an ablation only, the same as `int8_static` (same calibration data
  and settings) but with nothing excluded, so the input and feature block are quantised too.
  It is evaluated and benchmarked like the other variants.

Each run's `quant_report.json` records, per variant, the file size, op counts, which MatMul, Gemm,
Conv and GRU nodes stayed float, the checker and onnxruntime load results, val top-1, val top-1
agreement with fp32 and the largest logit difference from fp32 on 512 val sequences. It also
records `excluded_node_count` and `excluded_node_names` (the nodes passed to the quantiser's
`nodes_to_exclude`, empty for every variant except `int8_static`) and `input_quantised`, whether a
QuantizeLinear or DynamicQuantizeLinear node reads `landmarks`. A variant that fails is recorded
there and the other variants still run. `results/tables/quant_ops_summary.csv` summarises the
reports. Existing variant files are reused unless `--overwrite` is passed; a file built from a
different `best.pt` is reported as stale. Each file records the schema version of its variant's
build recipe, and a file from an older version is rebuilt without `--overwrite`: `int8_static`
files built before the feature-block exclusion are replaced this way, while the other variants'
files are kept.

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
`transformer` at seed 0, variants `torch_fp32`, `onnx_fp32`, `fp16`, `int8_dynamic`,
`int8_static` and `int8_static_full` (the ablation), and 1, 2 and 4 intra-op threads. `--batch N`
switches to a throughput mode whose output files carry a `_batch<N>` suffix.

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
  about 8 minutes.
- **Machine state.** Run on AC power, with the Windows power mode set to Best performance and
  other applications closed. `results/bench/bench_env.json` records the CPU flags detected at
  runtime (AVX2, AVX-512, VNNI), the clock at the start and end, the power source, the active
  power scheme and power mode, and the library versions. A run on battery prints a warning and
  records it there.

Outputs: `results/bench/bench_raw.csv` (one row per configuration and round),
`results/bench/bench_env.json` and `results/tables/bench_summary.csv`. The summary's
`p95_share_of_33ms_frame` column is p95 latency as a share of one 30 fps frame.

## Web demo

Live: <https://swapnil007-curious.github.io/edge-vision/>

`web/` is an installable, offline-capable browser app (Vite and strict TypeScript, no UI
framework) that recognises one isolated sign at a time from the webcam. MediaPipe Tasks
HandLandmarker (two hands) and PoseLandmarker lite produce landmarks for each frame, a TypeScript
port of the Python preprocessing (`web/src/preprocess.ts`) turns each captured segment into the
model input, and onnxruntime-web runs the exported classifier. Every step runs in the browser.

- **Scope.** Recognises single isolated signs from a 250-sign ASL vocabulary (the asl-signs
  labels). It does not translate sentences.
- **Privacy.** Video never leaves the device. There is no analytics or telemetry, and after the
  first load the app makes no network requests: a Content-Security-Policy restricts every request
  to the app's own origin, and a service worker serves the app, the model and the MediaPipe files
  from the device. Spoken results, which are off by default, use on-device voices only.

### Run locally

Node 20.19 or newer, from `web/`:

```powershell
npm ci
npm run fetch-assets   # MediaPipe wasm and .task models into public/mediapipe/ (not in git)
npm run dev            # development server
npm run build          # production build into dist/
npm run preview        # serves dist/ at http://localhost:4173/
```

Camera access needs a secure context; `localhost` qualifies. `npm run fetch-assets` checks every
file against a pinned or npm-provided sha256, writes `public/mediapipe/ASSETS.json` (URL, sha256,
size, licence) and skips files that are already present.

Tests: `npx tsc --noEmit`, `npx vitest run` (preprocessing against the Python golden vectors,
segmentation, scoring, settings) and, after `npm run build`, `npx playwright install chromium`
then `npx playwright test` (model and preprocessing parity in Chromium, offline reload through
the service worker, and a camera smoke test with Chromium's fake camera). For a build under
another base path, set `BASE_PATH` (for example `/edge-vision/`) for both the build and the tests.

### Capture

In automatic mode a segment starts after 3 consecutive frames with a hand and ends after 8
frames without one, or after 3.0 s. Segments with fewer hand frames than the model's
`min_hand_frames` are discarded. In "hold to record" mode, hold Space or press and hold the large
button while signing. If the top softmax probability reaches the threshold stored in `meta.json`,
the sign is shown; otherwise the app shows "Not sure" with the three most likely signs. All
thresholds can be changed under Settings; settings are kept in `localStorage` when the browser
allows it. Framing can be the full camera frame or a centred 3:4 portrait crop, applied before
MediaPipe. Guidance messages report missing hands, hands partly out of frame, low light, and a
signer too close or too far (shoulder distance in normalised units).

### Exporting a model

`scripts/export_web.py` copies one quantised variant to `web/public/model/model.onnx` and writes
`web/public/model/meta.json` (labels, input layout, landmark order, preprocessing settings,
sha256 of the ONNX file and of `best.pt`, opset, and the confidence threshold), plus the golden
vectors in `web/tests/golden/`:

```powershell
python scripts/export_web.py --run results/runs/<model>_signer_independent/seed_0 --variant int8_static --quant-root results/quant --processed-dir data/processed
```

- **Variant rule.** Use `int8_static` if its val top-1 (from `quant_report.json`) is within 1.0
  percentage point of fp32 val top-1, otherwise `fp32`. The script prints the comparison;
  `--variant` decides, and `meta.json` records both.
- **Confidence threshold.** Computed on val only, from the exported variant's softmax: the
  smallest threshold t at which top-1 accuracy among val sequences with max probability >= t
  reaches `--target-precision` (default 0.90). If no t reaches it, the t with the highest
  accuracy among those accepting at least 20% of val is used and `meta.json` records a warning.
  Coverage, accuracy above t and overall val top-1 are stored with it. The test split is never
  read.
- **Golden vectors.** `preprocess_cases.json` holds synthetic sequences built with the test
  suite's fixture builders (mirroring, ties, hand fallback, gaps, short and long sequences and
  both exclusion reasons) with the Python outputs; `model_cases.json` holds 16 processed val
  inputs and the logits Python onnxruntime gives for them. The Python tests check that the
  committed files still match the Python preprocessing and the committed model.
- `--data-source synthetic` marks fixture data in `meta.json`; the app then shows a banner that
  the model does not recognise real signs. `meta.json` always states which data the committed
  model came from.
- Commit `model.onnx` only while it is under 10 MB; `.gitignore` admits only that one ONNX file.

### In-browser benchmark

Start the camera, open Performance and press "Run 60-second benchmark", then sign a few times.
The panel shows live FPS and rolling p50/p95 over the last 300 samples for the hand landmarker,
pose landmarker, preprocessing, classifier and the total per frame. At the end a JSON file is
downloaded with the user agent, `hardwareConcurrency`, `deviceMemory` where available, the actual
camera resolution and frame rate, the settings, the model variant and sha256, raw per-frame and
per-segment timings, p50/p95 summaries, end-to-end FPS, and 50 classifier runs on a fixed input.
The camera is requested at 640 x 480 and 30 fps. The file is only saved locally.

### Deployment

`.github/workflows/pages.yml` builds `web/` with base `/edge-vision/` on every push to `main` and
deploys it with `actions/upload-pages-artifact` and `actions/deploy-pages`. In the repository,
Settings -> Pages -> Source must be set to "GitHub Actions". The `web` job in
`.github/workflows/ci.yml` runs the type check, unit tests, build and browser tests.

### Known limitations

- **Landmarks differ from the training data.** asl-signs was built with MediaPipe Holistic; the
  demo uses MediaPipe Tasks HandLandmarker and PoseLandmarker lite, whose landmarks, detection
  thresholds and failure modes are not identical, so accuracy in the browser can be lower than on
  the dataset.
- **Hand assignment.** Holistic's `left_hand` is the signer's own left hand. HandLandmarker labels
  handedness as if the image were mirrored, and the demo feeds it the unmirrored frame, so its
  "Left" label goes to `right_hand` and "Right" to `left_hand`. A "Swap hands" setting reverses
  this, and the Performance panel shows how often each hand sits at the matching pose wrist.
- **Camera aspect ratio.** Coordinates are normalised by the image width and height, so a camera
  whose aspect ratio differs from the recordings changes the geometry the model sees. The portrait
  crop narrows the gap for landscape webcams.
- **Isolated signs only.** Each segment is classified as one sign from the vocabulary; there is no
  sentence-level recognition, and signs outside the vocabulary are still mapped to one of its
  labels unless the confidence threshold rejects them.
- **Single-threaded inference.** GitHub Pages cannot send COOP/COEP headers, so onnxruntime-web
  runs without wasm threads. WebAssembly SIMD is required.

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
