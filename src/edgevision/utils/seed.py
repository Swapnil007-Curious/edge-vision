"""Seeding for reproducible runs."""

import importlib
import importlib.util
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed random, numpy and PYTHONHASHSEED, and torch when it is importable.

    PYTHONHASHSEED only takes effect in child processes; the running interpreter has already
    fixed its hash seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the legacy global state is what this must seed.
    if importlib.util.find_spec("torch") is not None:
        importlib.import_module("torch").manual_seed(seed)
