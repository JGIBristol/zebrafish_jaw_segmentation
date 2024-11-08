"""
Metrics for evaluating similarity etc. between images.

"""

import warnings

import numpy as np
from sklearn import metrics as skm


def _check_arrays(truth: np.ndarray, pred: np.ndarray) -> None:
    """
    Check the arrays are the right shape and formats

    """
    if truth.shape != pred.shape:
        raise ValueError(
            f"Shapes of truth {truth.shape} and pred {pred.shape} arrays do not match"
        )
    if set(np.unique(truth)) - {0, 1}:
        raise ValueError(f"truth array is not binary: {np.unique(truth)=}")
    if not (0 <= pred).all() and (pred <= 1).all():
        raise ValueError(
            f"pred array is not in the range [0, 1]: {np.min(pred)=}, {np.max(pred)=}"
        )


def dice_score(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the Dice score between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Dice score.
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    intersection = np.sum(truth * pred)
    volume1 = np.sum(truth)
    volume2 = np.sum(pred)

    # Both arrays are empty, consider Dice score as 1
    if volume1 + volume2 == 0:
        warnings.warn("Both arrays are empty, returning Dice score of 1")
        return 1.0

    return 2.0 * intersection / (volume1 + volume2)


def fpr(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the false positive rate between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: False positive rate.
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    inverse_negatives = 1 - truth
    negatives = np.sum(inverse_negatives)

    if negatives == 0:
        warnings.warn("No negatives; returning FPR of 0")
        return 0.0

    weighted_false_positives = np.sum(inverse_negatives * pred)

    return weighted_false_positives / negatives


def tpr(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the true positive rate between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: True positive rate.
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    positives = np.sum(truth)
    if positives == 0:
        warnings.warn("No positives; returning TPR of 0")
        return 0.0

    weighted_positives = np.sum(truth * pred)

    return weighted_positives / positives


def average_precision(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the precision, averaged over thresholds, between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Average precision score
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    return skm.average_precision_score(truth.flatten(), pred.flatten())


# recall
# Jaccard
# Hausdorff
# hausdorff profile
