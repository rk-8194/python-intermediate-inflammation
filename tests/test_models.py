"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_mean, load_json, daily_min, daily_max


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4])
    ],
)
def test_daily_mean(test, expected):
    """Test mean function works for an array of zeroe and positive integers.

    Args:
        test (np.array): Test array.
        expected (np.array): Mean of the test array.
    """
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
    ],
)
def test_daily_min(test, expected):
    """Test mean function works for an array of zeroe and positive integers.

    Args:
        test (np.array): Test array.
        expected (np.array): Mean of the test array.
    """
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
    ],
)
def test_daily_max(test, expected):
    """Test mean function works for an array of zeroe and positive integers.

    Args:
        test (np.array): Test array.
        expected (np.array): Mean of the test array.
    """
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


def test_load_from_json(tmpdir):
    """Test script to load from a json file.

    Args:
        tmpdir (string): Path to the json file.
    """
    example_path = os.path.join(tmpdir, "example.json")
    with open(example_path, "w", encoding="utf-8") as temp_json_file:
        temp_json_file.write('[{"observations":[1, 2, 3]},{"observations":[4, 5, 6]}]')
    result = load_json(example_path)
    npt.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])
