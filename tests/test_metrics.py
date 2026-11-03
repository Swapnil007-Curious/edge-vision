import numpy as np
import pytest

from edgevision.evaluation.metrics import error_pairs, macro_f1, per_group_accuracy, top_k_accuracy

LOGITS = np.array([[0.1, 0.5, 0.4], [0.9, 0.06, 0.04], [0.2, 0.3, 0.5]])
LABELS = np.array([2, 1, 2])


def test_top_k_accuracy():
    assert top_k_accuracy(LOGITS, LABELS, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(LOGITS, LABELS, 2) == pytest.approx(1.0)


def test_top_k_is_capped_at_the_class_count():
    assert top_k_accuracy(LOGITS, LABELS, 5) == 1.0


def test_macro_f1_by_hand():
    labels = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 1, 1, 1, 0])
    # Class F1: 0 -> 2*1/(2+2) = 0.5, 1 -> 2*2/(3+2) = 0.8, 2 -> 0.
    assert macro_f1(labels, predictions) == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_macro_f1_counts_classes_that_are_only_predicted():
    assert macro_f1(np.array([0, 0]), np.array([0, 1])) == pytest.approx((2 / 3 + 0.0) / 2)


def test_macro_f1_perfect_and_empty():
    labels = np.array([3, 1, 4, 1])
    assert macro_f1(labels, labels) == 1.0
    with pytest.raises(ValueError, match="empty"):
        macro_f1(np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def test_per_group_accuracy():
    result = per_group_accuracy(
        np.array([0, 1, 1, 2]), np.array([0, 0, 1, 2]), np.array([7, 7, 9, 9])
    )
    assert result == {7: (2, 0.5), 9: (2, 1.0)}


def test_error_pairs_count_only_mistakes():
    labels = np.array([0, 0, 0, 1, 2])
    predictions = np.array([1, 1, 0, 0, 2])
    assert error_pairs(labels, predictions) == {(0, 1): 2, (1, 0): 1}
    assert error_pairs(labels, labels) == {}
