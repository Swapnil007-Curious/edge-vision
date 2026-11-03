"""Calibration data for static quantisation, drawn from the train split and nothing else."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from edgevision.layout import INPUT_NAME
from edgevision.train.data import load_split_arrays

CALIBRATION_SAMPLES = 1024
CALIBRATION_SEED = 0
CALIBRATION_BATCH = 64


class TrainCalibrationReader:
    """Feeds onnxruntime's quantize_static ``n_samples`` train sequences in batches.

    The sequences are sampled without replacement by numpy Generator(seed) and kept in split
    order. The reader loads the train split itself and takes no split argument, so it cannot be
    pointed at val or test. onnxruntime accepts any object with get_next() as a
    CalibrationDataReader, so this class does not subclass it.
    """

    def __init__(
        self,
        processed_dir: Path,
        splits_path: Path,
        *,
        n_samples: int = CALIBRATION_SAMPLES,
        seed: int = CALIBRATION_SEED,
        batch_size: int = CALIBRATION_BATCH,
    ):
        train = load_split_arrays(processed_dir, splits_path, ["train"])["train"]
        rows = np.random.default_rng(seed).choice(
            len(train.y), size=min(n_samples, len(train.y)), replace=False
        )
        rows.sort()
        self.inputs: npt.NDArray[np.float32] = train.X[rows]
        self.sequence_ids: npt.NDArray[np.int64] = train.sequence_id[rows]
        self.train_sequences = len(train.y)
        self.seed = seed
        self.batch_size = batch_size
        self._position = 0

    def __len__(self) -> int:
        return len(self.inputs)

    def get_next(self) -> dict[str, npt.NDArray[np.float32]] | None:
        if self._position >= len(self.inputs):
            return None
        batch = self.inputs[self._position : self._position + self.batch_size]
        self._position += self.batch_size
        return {INPUT_NAME: batch}

    def rewind(self) -> None:
        self._position = 0
