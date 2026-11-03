"""Classification metrics in numpy, so evaluation needs no scikit-learn."""

from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


def top_k_accuracy(logits: npt.NDArray[Any], labels: npt.NDArray[Any], k: int) -> float:
    """Fraction of rows whose label is among the k highest logits; k is capped at class count."""
    if logits.shape[0] == 0:
        raise ValueError("cannot score an empty set")
    k = min(k, logits.shape[1])
    top = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return float((top == labels[:, None]).any(axis=1).mean())


def macro_f1(labels: npt.NDArray[Any], predictions: npt.NDArray[Any]) -> float:
    """Unweighted mean F1 over the classes present in labels or predictions.

    Matches scikit-learn's f1_score(average="macro") without an explicit ``labels`` argument.
    """
    if labels.size == 0:
        raise ValueError("cannot score an empty set")
    classes = np.union1d(labels, predictions)
    size = int(classes.max()) + 1
    true_positive = np.bincount(labels[labels == predictions], minlength=size)[classes]
    predicted = np.bincount(predictions, minlength=size)[classes]
    actual = np.bincount(labels, minlength=size)[classes]
    return float(np.mean(2.0 * true_positive / (predicted + actual)))


def per_group_accuracy(
    labels: npt.NDArray[Any], predictions: npt.NDArray[Any], groups: npt.NDArray[Any]
) -> dict[int, tuple[int, float]]:
    """Group id -> (number of rows, top-1 accuracy), e.g. per participant."""
    correct = labels == predictions
    return {
        int(group): (int((groups == group).sum()), float(correct[groups == group].mean()))
        for group in np.unique(groups)
    }


def error_pairs(
    labels: npt.NDArray[Any], predictions: npt.NDArray[Any]
) -> dict[tuple[int, int], int]:
    """(true label, predicted label) -> count, over misclassified rows only."""
    wrong = labels != predictions
    pairs, counts = np.unique(
        np.stack([labels[wrong], predictions[wrong]], axis=1), axis=0, return_counts=True
    )
    return {
        (int(true), int(pred)): int(count)
        for (true, pred), count in zip(pairs, counts, strict=True)
    }
