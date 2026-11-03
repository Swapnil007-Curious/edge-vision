"""Trained fixture runs for the quantisation, test-evaluation and benchmark tests.

Every architecture keeps its default size, so file-size ratios behave as they do on real models,
but trains for only 16 steps (4 epochs of 4 batches) on the synthetic data, with 3 train, 3 val
and 2 test signers so that val is large enough to measure agreement.
"""

from dataclasses import dataclass
from pathlib import Path

from edgevision.config import SplitConfig
from edgevision.models.config import MODEL_NAMES
from edgevision.runs import CompletedRun, find_completed_runs
from edgevision.train.config import AugmentConfig, load_train_config
from edgevision.train.loop import run_training
from training_support import EXPERIMENTS, SyntheticProcessed, build_processed_dir

FIXTURE_SPLIT = SplitConfig(n_test_signers=2, n_val_signers=3)
FIXTURE_EPOCHS = 4
FIXTURE_BATCH = 32
FIXTURE_LR = 3e-3
TRAIN_SEQUENCES, VAL_SEQUENCES, TEST_SEQUENCES = 120, 120, 80


@dataclass(frozen=True)
class FixtureRuns:
    processed: SyntheticProcessed
    runs_root: Path

    def run(self, model: str) -> CompletedRun:
        (match,) = [
            run for run in find_completed_runs(self.runs_root) if run.summary["model"] == model
        ]
        return match


def train_fixture_runs(root: Path) -> FixtureRuns:
    """One signer-independent seed-0 run per architecture, in the standard run layout."""
    processed = build_processed_dir(root / "processed", write_splits=False)
    runs_root = root / "runs"
    for model in MODEL_NAMES:
        cfg = load_train_config(
            EXPERIMENTS / f"train_{model}.yaml",
            processed_dir=processed.root,
            output_root=runs_root,
            epochs=FIXTURE_EPOCHS,
        ).model_copy(
            update={
                "batch_size": FIXTURE_BATCH,
                "lr": FIXTURE_LR,
                "split": FIXTURE_SPLIT,
                "augment": AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=0.0),
            }
        )
        run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    return FixtureRuns(processed, runs_root)
