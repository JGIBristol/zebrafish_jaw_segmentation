"""
Metrics for evaluating similarity etc. between images.

"""

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from skimage import metrics as skimage_m


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
    Calculate the precision, averaged over thresholds,
    between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Average precision score
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    return skm.average_precision_score(truth.flatten(), pred.flatten())


def recall(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the recall score between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Average precision score
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    return skm.recall_score(truth.flatten(), pred.flatten())


def jaccard(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the Jaccard score between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Jaccard score
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    intersection = np.sum(truth * pred)
    union = np.sum(truth + pred)

    return intersection / union


def roc_auc(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the ROC AUC score between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: ROC AUC score
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    _check_arrays(truth, pred)

    return skm.roc_auc_score(truth.flatten(), pred.flatten())


def _check_arrays_binary(truth: np.ndarray, pred: np.ndarray) -> None:
    """
    Check the arrays are the right shape and formats

    """
    if truth.shape != pred.shape:
        raise ValueError(
            f"Shapes of truth {truth.shape} and pred {pred.shape} arrays do not match"
        )
    if set(np.unique(truth)) - {0, 1}:
        raise ValueError(f"truth array is not binary: {np.unique(truth)=}")
    if set(np.unique(pred)) - {0, 1}:
        raise ValueError(f"prediction array is not binary: {np.unique(pred)=}")


def hausdorff_distance(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between a binary mask (truth) and a binary array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Hausdorff distance
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.

    """
    _check_arrays_binary(truth, pred)

    return skimage_m.hausdorff_distance(truth, pred)


def hausdorff_profile(
    truth: np.ndarray, pred: np.ndarray, thresholds: Iterable = None
) -> list[float]:
    """
    Calculate the hausdorff distance for a range of thresholds between a binary mask
    (truth) and a binary array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.
    :param thresholds: Thresholds to use for the prediction array. If None, uses 50 thresholds
        between 0 and 1.

    :returns: Hausdorff distance for each threshold
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.

    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 50, endpoint=True)

    return [hausdorff_distance(truth, pred > threshold) for threshold in thresholds]


def hausdorff_dice(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the combined Hausdorff-Dice metric between a binary mask (truth)
    and a binary array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Hausdorff-Dice distance
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.

    """
    _check_arrays_binary(truth, pred)

    # Scale the hausdorff distance to the maximum possible value (i.e. the diagonal)
    scaled_distance = hausdorff_distance(truth, pred) / np.sqrt(
        np.sum(a**2 for a in truth.shape)
    )

    return scaled_distance + (1 - dice_score(truth, pred))


def table(truth: list[np.ndarray], pred: list[np.ndarray]) -> pd.DataFrame:
    """
    Return a table of metrics between a binary mask (truth) and a float array (pred)
    in a nice markdown format

    :param truth: List of binary mask arrays.
    :param pred: List of float prediction arrays.

    :returns: Table of metrics

    """
    df = pd.DataFrame()

    df["Dice"] = [dice_score(t, p) for t, p in zip(truth, pred)]
    df["FPR"] = [fpr(t, p) for t, p in zip(truth, pred)]
    df["TPR"] = [tpr(t, p) for t, p in zip(truth, pred)]
    df["Precision"] = [average_precision(t, p) for t, p in zip(truth, pred)]
    # df["Recall"] = [recall(t, p) for t, p in zip(truth, pred)]
    df["Jaccard"] = [jaccard(t, p) for t, p in zip(truth, pred)]
    df["ROC AUC"] = [roc_auc(t, p) for t, p in zip(truth, pred)]

    # Threshold the prediction
    thresholds = (0.5,)
    for threshold in thresholds:
        df[f"Hausdorff_{threshold}"] = [
            hausdorff_distance(t, p > threshold) for t, p in zip(truth, pred)
        ]

    return df
