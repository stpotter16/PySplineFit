'''

    Tests for knots module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2018 Sam Potter

    Requires pytest
'''

from .context import pysplinefit
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
