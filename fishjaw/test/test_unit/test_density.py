"""
Tests for the density analysis stuff
"""

import numpy as np
from scipy.spatial.distance import squareform

from fishjaw.density_analysis import find_attachments


def test_pairwise_distance():
    """
    Check the pairwise distance fcn does what I expect
    """
    points = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([2, 0]),
        np.array([1, 1]),
    ]

    root2 = np.sqrt(2)
    root5 = np.sqrt(5)
    expected = squareform(
        np.array(
            [
                [0, 1, 2, root2],
                [1, 0, root5, 1],
                [2, root5, 0, root2],
                [root2, 1, root2, 0],
            ]
        )
    )

    np.testing.assert_array_almost_equal(
        find_attachments.get_pairwise_distances(points), expected
    )
