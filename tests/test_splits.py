import numpy as np
import pandas as pd
import pytest

from edgevision.config import SplitConfig
from edgevision.data.splits import (
    assert_signer_disjoint,
    make_splits,
    splits_to_json,
    summarize_splits,
)


def make_manifest(n_participants=10, n_labels=5, per_cell=4, *, excluded=None):
    rows = []
    for participant in range(n_participants):
        for label in range(n_labels):
            for repeat in range(per_cell):
                rows.append(
                    {
                        "sequence_id": 100_000 * participant + 100 * label + repeat,
                        # Above int16 on purpose: ids must round-trip unchanged.
                        "participant_id": 40_000 + participant,
                        "sign": f"sign{label}",
                        "label": label,
                        "excluded": False,
                    }
                )
    manifest = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)
    if excluded is not None:
        manifest["excluded"] = excluded(manifest)
    return manifest


def owners(manifest, sequence_ids):
    table = manifest.set_index("sequence_id")["participant_id"]
    return set(table.loc[sequence_ids.tolist()].tolist())


def test_signer_independent_splits_are_disjoint_and_complete():
    manifest = make_manifest()
    splits = make_splits(manifest, SplitConfig())

    train, val, test = (owners(manifest, splits[key]) for key in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (3, 3, 4)
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(manifest["participant_id"])
    assert splits["train_participants"] == sorted(train)
    assert splits["val_participants"] == sorted(val)
    assert splits["test_participants"] == sorted(test)
    combined = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert sorted(combined.tolist()) == sorted(manifest["sequence_id"].tolist())
    assert_signer_disjoint(splits, manifest)


def test_signer_independent_is_deterministic_for_a_seed():
    manifest = make_manifest()
    first = make_splits(manifest, SplitConfig(seed=5))
    # Row order must not leak into the assignment.
    second = make_splits(manifest.sample(frac=1.0, random_state=1), SplitConfig(seed=5))
    for key in ("train", "val", "test"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["test_participants"] == second["test_participants"]


def test_signer_independent_differs_for_another_seed():
    manifest = make_manifest()
    first = make_splits(manifest, SplitConfig(seed=42))
    second = make_splits(manifest, SplitConfig(seed=43))
    assert first["test_participants"] != second["test_participants"]


def test_signer_independent_needs_enough_participants():
    with pytest.raises(ValueError, match="at least 8 participants"):
        make_splits(make_manifest(n_participants=7), SplitConfig())
    assert make_splits(make_manifest(n_participants=8), SplitConfig())


def test_excluded_sequences_are_not_split():
    manifest = make_manifest(excluded=lambda frame: frame["sequence_id"] % 2 == 1)
    splits = make_splits(manifest, SplitConfig())
    combined = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert sorted(combined.tolist()) == sorted(
        manifest.loc[~manifest["excluded"], "sequence_id"].tolist()
    )


def test_fully_excluded_participant_does_not_count():
    manifest = make_manifest(
        n_participants=8, excluded=lambda frame: frame["participant_id"] == 40_000
    )
    with pytest.raises(ValueError, match="found 7"):
        make_splits(manifest, SplitConfig())


def test_assert_signer_disjoint_names_injected_overlap():
    manifest = make_manifest()
    splits = make_splits(manifest, SplitConfig())
    leaked = splits["train"][0]
    leaked_owner = manifest.loc[manifest["sequence_id"] == leaked, "participant_id"].item()
    splits["test"] = np.append(splits["test"], leaked)
    with pytest.raises(AssertionError, match=rf"train and test.*{leaked_owner}"):
        assert_signer_disjoint(splits, manifest)


def test_assert_signer_disjoint_uses_sequences_not_stored_lists():
    manifest = make_manifest()
    splits = make_splits(manifest, SplitConfig())
    splits["val"] = np.append(splits["val"], splits["test"][0])
    splits["val_participants"] = []
    with pytest.raises(AssertionError, match="val and test"):
        assert_signer_disjoint(splits, manifest)


def test_assert_signer_disjoint_rejects_unknown_sequences():
    manifest = make_manifest()
    splits = make_splits(manifest, SplitConfig())
    splits["val"] = np.append(splits["val"], -1)
    with pytest.raises(AssertionError, match="not in the manifest"):
        assert_signer_disjoint(splits, manifest)


def test_signer_dependent_is_stratified_seeded_and_complete():
    manifest = make_manifest(n_participants=5, per_cell=20)
    cfg = SplitConfig(mode="signer_dependent", val_fraction=0.1, test_fraction=0.2, seed=3)
    splits = make_splits(manifest, cfg)

    labels = manifest.set_index("sequence_id")["label"]
    for label in range(5):
        assert (labels.loc[splits["val"].tolist()] == label).sum() == 10
        assert (labels.loc[splits["test"].tolist()] == label).sum() == 20
        assert (labels.loc[splits["train"].tolist()] == label).sum() == 70

    combined = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert sorted(combined.tolist()) == sorted(manifest["sequence_id"].tolist())
    assert "train_participants" not in splits

    repeat = make_splits(manifest.sample(frac=1.0, random_state=2), cfg)
    np.testing.assert_array_equal(splits["test"], repeat["test"])
    other = make_splits(manifest, cfg.model_copy(update={"seed": 4}))
    assert not np.array_equal(splits["test"], other["test"])


def test_summary_counts_sequences_and_labels():
    manifest = make_manifest()
    splits = make_splits(manifest, SplitConfig())
    summary = summarize_splits(splits, manifest)
    assert summary == {
        "train": {"sequences": 60, "labels": 5},
        "val": {"sequences": 60, "labels": 5},
        "test": {"sequences": 80, "labels": 5},
    }


def test_summary_warns_when_a_label_is_missing_from_val_or_test():
    manifest = make_manifest()
    splits = make_splits(manifest, SplitConfig())
    test_owner = splits["test_participants"]
    mask = manifest["participant_id"].isin(test_owner) & (manifest["label"] == 2)
    manifest = manifest.loc[~mask].reset_index(drop=True)
    splits = make_splits(manifest, SplitConfig())
    with pytest.warns(UserWarning, match="labels missing from test"):
        summary = summarize_splits(splits, manifest)
    assert summary["test"]["labels"] == 4


def test_splits_json_payload_is_serialisable():
    import json

    manifest = make_manifest()
    cfg = SplitConfig()
    splits = make_splits(manifest, cfg)
    payload = splits_to_json(splits, summarize_splits(splits, manifest), cfg)
    decoded = json.loads(json.dumps(payload))
    assert decoded["mode"] == "signer_independent"
    assert decoded["participants"]["test"] == splits["test_participants"]
    assert decoded["train"] == splits["train"].tolist()
