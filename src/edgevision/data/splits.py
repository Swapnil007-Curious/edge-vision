"""Train/val/test splits over the preprocessing manifest.

Only non-excluded rows are split, so every sequence_id returned exists in features.npz.
"""

import itertools
import warnings
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.config import SplitConfig

IdArray = npt.NDArray[np.int64]
SplitName = Literal["train", "val", "test"]

SPLIT_NAMES: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")


class Splits(TypedDict):
    train: IdArray
    val: IdArray
    test: IdArray
    train_participants: NotRequired[list[int]]
    val_participants: NotRequired[list[int]]
    test_participants: NotRequired[list[int]]


def _kept_rows(manifest_df: pd.DataFrame) -> pd.DataFrame:
    return manifest_df.loc[~manifest_df["excluded"].astype(bool)]


def _sequences_of(kept: pd.DataFrame, participants: npt.NDArray[np.int64]) -> IdArray:
    rows = kept["participant_id"].isin(participants.tolist())
    return np.sort(kept.loc[rows, "sequence_id"].to_numpy(dtype=np.int64))


def _signer_independent(kept: pd.DataFrame, cfg: SplitConfig) -> Splits:
    participants = np.unique(kept["participant_id"].to_numpy(dtype=np.int64))  # Sorted.
    n_test, n_val = cfg.n_test_signers, cfg.n_val_signers
    needed = n_test + n_val + 1
    if participants.size < needed:
        raise ValueError(
            f"signer_independent needs at least {needed} participants with kept sequences, "
            f"found {participants.size}"
        )

    shuffled = np.random.default_rng(cfg.seed).permutation(participants)
    test_ids = shuffled[:n_test]
    val_ids = shuffled[n_test : n_test + n_val]
    train_ids = shuffled[n_test + n_val :]
    return Splits(
        train=_sequences_of(kept, train_ids),
        val=_sequences_of(kept, val_ids),
        test=_sequences_of(kept, test_ids),
        train_participants=sorted(train_ids.tolist()),
        val_participants=sorted(val_ids.tolist()),
        test_participants=sorted(test_ids.tolist()),
    )


def _signer_dependent(kept: pd.DataFrame, cfg: SplitConfig) -> Splits:
    """Stratified random split; signers overlap, so this is only for ablation."""
    if kept.empty:
        raise ValueError("signer_dependent needs at least one kept sequence")
    rng = np.random.default_rng(cfg.seed)
    parts: dict[SplitName, list[IdArray]] = {name: [] for name in SPLIT_NAMES}
    for _, group in kept.sort_values(["label", "sequence_id"]).groupby("label", sort=True):
        sequences = rng.permutation(group["sequence_id"].to_numpy(dtype=np.int64))
        n_test = round(sequences.size * cfg.test_fraction)
        n_val = round(sequences.size * cfg.val_fraction)
        parts["test"].append(sequences[:n_test])
        parts["val"].append(sequences[n_test : n_test + n_val])
        parts["train"].append(sequences[n_test + n_val :])
    return Splits(
        train=np.sort(np.concatenate(parts["train"])),
        val=np.sort(np.concatenate(parts["val"])),
        test=np.sort(np.concatenate(parts["test"])),
    )


def make_splits(manifest_df: pd.DataFrame, cfg: SplitConfig) -> Splits:
    kept = _kept_rows(manifest_df)
    if cfg.mode == "signer_independent":
        return _signer_independent(kept, cfg)
    return _signer_dependent(kept, cfg)


def assert_signer_disjoint(splits: Splits, manifest_df: pd.DataFrame) -> None:
    """Check disjointness from the sequence ids themselves, not the stored participant lists."""
    owners = manifest_df.set_index("sequence_id")["participant_id"]
    signers: dict[SplitName, set[int]] = {}
    for name in SPLIT_NAMES:
        sequence_ids = splits[name].tolist()
        unknown = sorted(set(sequence_ids) - set(owners.index.tolist()))
        if unknown:
            raise AssertionError(f"{name} holds sequence ids not in the manifest: {unknown[:10]}")
        signers[name] = set(owners.loc[sequence_ids].astype(int).tolist())

    for first, second in itertools.combinations(SPLIT_NAMES, 2):
        shared = sorted(signers[first] & signers[second])
        if shared:
            raise AssertionError(f"participants in both {first} and {second}: {shared}")


def summarize_splits(splits: Splits, manifest_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-split sequence and label counts; warns if a label is missing from val or test."""
    kept = _kept_rows(manifest_df)
    labels_by_sequence = kept.set_index("sequence_id")["label"]
    all_labels = set(kept["label"].astype(int).tolist())

    summary: dict[str, dict[str, int]] = {}
    for name in SPLIT_NAMES:
        present = set(labels_by_sequence.loc[splits[name].tolist()].astype(int).tolist())
        summary[name] = {"sequences": int(splits[name].size), "labels": len(present)}
        missing = sorted(all_labels - present)
        if name != "train" and missing:
            warnings.warn(
                f"{len(missing)} labels missing from {name}: {missing[:10]}",
                stacklevel=2,
            )
    return summary


def splits_to_json(
    splits: Splits, summary: dict[str, dict[str, int]], cfg: SplitConfig
) -> dict[str, Any]:
    """JSON-ready view of the splits, for writing to splits_<mode>.json."""
    payload: dict[str, Any] = {"mode": cfg.mode, "seed": cfg.seed, "summary": summary}
    for name in SPLIT_NAMES:
        payload[name] = splits[name].tolist()
    if cfg.mode == "signer_independent":
        payload["participants"] = {
            "train": splits.get("train_participants"),
            "val": splits.get("val_participants"),
            "test": splits.get("test_participants"),
        }
    return payload
