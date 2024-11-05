"""
Fish metadata

"""

import numpy as np

from ..images import transform


def age(n: int) -> float:
    """
    Get the age of a fish in months.

    :param n: fish number, using Wahab's "new n" convention (the same one as in jaw_centres.csv).

    :returns: the age of the fish in months

    """
    age_ = transform.jaw_centres().loc[n, "age"]

    if not isinstance(age, np.float64):
        raise ValueError(f"Got multiple ages for fish {n}:\n\t{age_}")

    return age_
