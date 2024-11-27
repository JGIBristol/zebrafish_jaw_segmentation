"""
Metrics for evaluating similarity etc. between images.

"""

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage
from numpy.typing import NDArray
from sklearn import metrics as skm
from skimage import metrics as skimage_m


def _check_arrays(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> None:
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


def dice_score(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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

    # Both arrays are empty, consider Dice score to be 1
    if volume1 + volume2 == 0:
        warnings.warn("Both arrays are empty, returning Dice score of 1")
        return 1.0

    return 2.0 * intersection / (volume1 + volume2)


def float_dice(arr1: NDArray[np.float32], arr2: NDArray[np.float32]) -> float:
    """
    Calculate the Dice score between two float arrays.

    :param arr1: float array.
    :param arr2: float array.

    :returns: "Dice" score - actually just the product divided by the sum
    :raises: ValueError if the shapes of the arrays do not match.

    """
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape=} and {arr2.shape=}")

    intersection = np.sum(arr1 * arr2)
    volume1 = np.sum(arr1)
    volume2 = np.sum(arr2)

    # Both arrays are empty, consider Dice score to be 1
    if volume1 + volume2 == 0:
        warnings.warn("Both arrays are empty, returning Dice score of 1")
        return 1.0

    return 2.0 * intersection / (volume1 + volume2)


def fpr(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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


def tpr(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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


def precision(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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

    return np.sum(truth * pred) / np.sum(pred)


def recall(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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

    # True positives / all positives
    return np.sum(truth * pred) / np.sum(truth)


def g_measure(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
    """
    Calculate the geometric mean of the precision and recall

    :param truth: Binary mask array.
    :param pred: Float prediction array.
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the pred array is not in the range [0, 1].

    """
    return np.sqrt(precision(truth, pred) * recall(truth, pred))


def jaccard(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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
    union = np.sum(truth + pred) - intersection

    return intersection / union


def roc_auc(truth: NDArray[np.uint8], pred: NDArray[np.float32]) -> float:
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
    try:
        return skm.roc_auc_score(truth.flatten(), pred.flatten())
    except ValueError:
        warnings.warn("ROC AUC score could not be calculated")
        return float("nan")


def _check_arrays_binary(truth: NDArray[np.uint8], pred: NDArray[np.uint8]) -> None:
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


def hausdorff_points(
    truth: NDArray[np.uint8], pred: NDArray[np.uint8]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Find the points of the binary mask (truth) and a binary array (pred) that are
    separated by the two-directional Hausdorff distance.

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Array representing xyz of one point
    :returns: Array representing xyz of the other point

    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.

    """
    _check_arrays_binary(truth, pred)

    return skimage_m.hausdorff_pair(truth, pred)


def hausdorff_distance(truth: NDArray[np.uint8], pred: NDArray[np.uint8]) -> float:
    """
    Calculate the Hausdorff distance between a binary mask (truth) and a binary array (pred),
    scaled by the image dimensions.

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Hausdorff distance
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.

    """
    _check_arrays_binary(truth, pred)

    h_d = skimage_m.hausdorff_distance(truth, pred)

    # Scale by the image dimensions
    return h_d / np.sqrt(np.sum(a**2 for a in truth.shape))


def hausdorff_profile(
    truth: NDArray[np.uint8],
    pred: NDArray[np.float32],
    thresholds: Iterable[float] | None = None,
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


def hausdorff_dice(truth: NDArray, pred: NDArray, k: float = 0.25) -> float:
    """
    Calculate the combined Hausdorff-Dice metric between a binary mask (truth)
    and a binary array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.
    :param k: Weighting factor for the Dice score.

    :returns: Hausdorff-Dice distance
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.
    :raises: ValueError if k is not between 0 and 1

    """
    _check_arrays_binary(truth, pred)

    if not 0 < k < 1:
        raise ValueError(f"Expected k to be between 0 and 1, got {k}")

    return k * dice_score(truth, pred) + (1 - k) * (1 - hausdorff_distance(truth, pred))


def z_distance_score(truth: NDArray, pred: NDArray) -> float:
    """
    Calculate the weighted Z-distance between a binary mask (truth) and a float array (pred).

    :param truth: Binary mask array.
    :param pred: Float prediction array.

    :returns: Z-distance
    :raises: ValueError if the shapes of the arrays do not match.
    :raises: ValueError if the truth array is not binary.
    :raises: ValueError if the prediction array is not binary.

    """
    _check_arrays(truth, pred)

    truth_slice_sums = np.sum(truth, axis=(1, 2))
    pred_slice_sums = np.sum(pred, axis=(1, 2))

    numerator = np.sum((truth_slice_sums - pred_slice_sums) ** 2)
    denominator = np.sum(truth_slice_sums**2) + np.sum(pred_slice_sums**2)

    return 1 - numerator / denominator


def largest_connected_component(binary_array: NDArray) -> NDArray:
    """
    Return the largest connected component of a binary array, as a binary array,
    using a 26-connectivity.

    :param binary_array: Binary array.
    :returns: Largest connected component.

    """
    labelled, _ = ndimage.label(binary_array, np.ones((3, 3, 3)))

    # Find the size of each component, ignoring the background
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0

    return labelled == np.argmax(sizes)


def table(
    truth: list[NDArray[np.uint8]],
    pred: list[NDArray[np.float32]],
    thresholded_metrics: bool = False,
) -> pd.DataFrame:
    """
    Return a table of metrics between a binary mask (truth) and a float array (pred)
    in a nice markdown format

    :param truth: List of binary mask arrays.
    :param pred: List of float prediction arrays.
    :param thresholded_metrics: Whether to add some metrics for thresholded predictions.

    :returns: Table of metrics

    """
    df = pd.DataFrame()

    df["Dice"] = [dice_score(t, p) for t, p in zip(truth, pred)]
    df["Jaccard"] = [jaccard(t, p) for t, p in zip(truth, pred)]

    # I don't care about these
    # df["Z_dist_score"] = [z_distance_score(t, p) for t, p in zip(truth, pred)]
    # df["1-FPR"] = [1 - fpr(t, p) for t, p in zip(truth, pred)]
    # df["TPR"] = [tpr(t, p) for t, p in zip(truth, pred)]
    # df["Precision"] = [precision(t, p) for t, p in zip(truth, pred)]
    # df["Recall"] = [recall(t, p) for t, p in zip(truth, pred)]
    # df["ROC AUC"] = [roc_auc(t, p) for t, p in zip(truth, pred)]
    # df["G_Measure"] = [g_measure(t, p) for t, p in zip(truth, pred)]

    # Threshold the prediction
    if thresholded_metrics:
        thresholds = (0.5,)
        for threshold in thresholds:
            hd = []
            hd_dice = []
            for t, p in zip(truth, pred):
                thresholded = p > threshold

                # I don't actually think I want to take the largest component - for now
                # thresholded = largest_connected_component(thresholded)

                hd.append(1 - hausdorff_distance(t, thresholded))

                # A bit inefficient but keeps the implementation in one place
                hd_dice.append(hausdorff_dice(t, thresholded))

            df[f"1-Hausdorff_{threshold}"] = hd
            df[f"Hausdorff_Dice_{threshold}"] = hd_dice

    return df
