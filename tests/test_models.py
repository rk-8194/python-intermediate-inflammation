"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_mean, load_json, daily_min, daily_max


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_min_zero():
    """Test script to get the minimum from the data set."""
    test_input = np.array([[0, 0], [1, 1], [1, 1]])
    test_result = np.array([0, 0])

    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min():
    """Test script to get the minimum from the data set."""
    test_input = np.array([[1, 2], [3, 4], [5, 0]])
    test_result = np.array([1, 0])

    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_max_zero():
    """Test script to get the maximum from the data set."""
    test_input = np.array([[1, 1], [0, 0], [0, 0]])
    test_result = np.array([1, 1])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    with pytest.raises(TypeError):
        error_expected = daily_min([["Hello", "there"], ["General", "Kenobi"]])
        print(error_expected)


def test_daily_max():
    """Test script to get the minimum from the data set."""
    test_input = np.array([[1, 2], [3, 4], [5, 0]])
    test_result = np.array([5, 4])

    npt.assert_array_equal(daily_max(test_input), test_result)


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
