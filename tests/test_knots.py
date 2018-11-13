'''

    Tests for knots module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2018 Sam Potter

    Requires pytest
'''

from .context import pysplinefit
from pysplinefit import knots

import pytest


def test_find_span():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    num_ctrlps = len(knot_vector) - degree - 1
    knot = 5.0 / 2.0

    interval = knots.find_span(num_ctrlps, degree, knot, knot_vector)

    # NURBS Book Example 2.3
    correct = 4

    assert interval == correct


