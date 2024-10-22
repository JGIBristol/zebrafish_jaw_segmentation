""" Tests for data related utilities """

from ...model import data


def test_get_transforms():
    """
    Check we can get a torchio transform from a str and dict of args

    """
    # Check we can get a randomflip
    transform = data.load_transform(
        "torchio.RandomFlip", {"flip_probability": 0.5, "axes": [0, 1, 2]}
    )
    assert transform.flip_probability == 0.5
    assert transform.axes == (0, 1, 2)

    # Check we can get random affine
    transform = data.load_transform(
        "torchio.RandomAffine", {"p": 0.25, "degrees": 10, "scales": 0.2}
    )
    assert transform.probability == 0.25
    assert transform.degrees == (-10, 10, -10, 10, -10, 10)
    assert transform.scales == (0.8, 1.2, 0.8, 1.2, 0.8, 1.2)
