"""
Unit tests for util module

"""

import pytest

from ...util import util


def test_call_once() -> None:
    """
    Check that a function can only be called once

    """

    @util.call_once
    def test_func() -> None:
        """Test function"""
        pass

    test_func()
    with pytest.raises(RuntimeError):
        test_func()
