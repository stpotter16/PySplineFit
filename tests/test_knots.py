"""

    Tests for knots module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from .context import np
from pysplinefit import knots

import pytest


@pytest.mark.parametrize('knot_val, expected',
                         [
                             (5.0 / 2.0, 4),
                             (1.0 / 2.0, 2),
                             (7.0 / 2.0, 5),
                             (9.0 / 2.0, 7),
                             (3.0, 5),
                             (0.0, 2),  # Edge case: Should return left side of interval
                             (3.0, 5),  # Should return left side of interval
                             (5.0, 7),  # Edge case: should return num_ctrlpts - 1 (n)
                         ]
                         )
def test_find_span(knot_val, expected):
    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    num_ctrlps = len(knot_vector) - degree - 1

    interval = knots.find_span(num_ctrlps, degree, knot_val, knot_vector)

    assert interval == expected


def test_normalize():
    array = np.array([0, 0, 0, 1, 2, 2, 2])
    expected = np.array([0, 0, 0, 0.5, 1, 1, 1])

    normalized = knots.normalize(array)

    condition = np.allclose(normalized, expected)

    assert condition


def test_check():
    degree = 2
    knot_vector = [0, 0, 0, 0.5, 1, 1, 1]
    num_ctrlpts = 4

    check_val = knots.check_knot_vector(degree, knot_vector, num_ctrlpts)

    expected = True

    assert check_val == expected


def test_check2():
    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 2, 2]
    num_ctrlpts = 4

    check_val = knots.check_knot_vector(degree, knot_vector, num_ctrlpts)

    expected = True

    assert check_val == expected


def test_check3():
    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 2, 2]
    num_ctrlpts = 5

    check_val = knots.check_knot_vector(degree, knot_vector, num_ctrlpts)

    expected = False

    assert check_val == expected


def test_generate():
    degree = 2
    num_ctrlpts = 4

    expected = np.array([0, 0, 0, 0.5, 1, 1, 1])

    generated = knots.generate_uniform(degree, num_ctrlpts)

    condition = np.allclose(generated, expected)

    assert condition


def test_generate2():
    degree = 3
    num_ctrlpts = 4

    expected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    generated = knots.generate_uniform(degree, num_ctrlpts)

    condition = np.allclose(generated, expected)

    assert condition


def test_generate3():
    degree = 2
    num_ctrlpts = 6

    expected = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])

    generated = knots.generate_uniform(degree, num_ctrlpts)

    condition = np.allclose(generated, expected)

    assert condition


def test_multiplicity():

    knot_vector = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])

    knot = 0.625

    expected = 0

    mult_val = knots.find_multiplicity(knot, knot_vector)

    assert mult_val == expected


def test_multiplicity2():

    knot_vector = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])

    knot = 0.5

    expected = 1

    mult_val = knots.find_multiplicity(knot, knot_vector)

    assert mult_val == expected


def test_multiplicity3():

    knot_vector = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])

    knot = 0.0

    expected = 3

    mult_val = knots.find_multiplicity(knot, knot_vector)

    assert mult_val == expected
