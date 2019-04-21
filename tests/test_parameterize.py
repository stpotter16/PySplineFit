"""

    Tests for parameterize module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from .context import np
from pysplinefit import spline
from pysplinefit import parameterize

import pytest


@pytest.fixture
def curve():
    """ Generate curve for test """
    curve = spline.Curve()
    curve.degree = 3
    curve.control_points = np.array([[0, 0, 0],
                                     [1, 2, 0],
                                     [2, 2, 0],
                                     [3, 0, 0]])
    curve.knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    return curve


def test_initial_guess_curve(curve):
    eval_knot = 1 / (2 * len(curve.control_points) - 1) * 2
    eval_point = curve.single_point(eval_knot)

    test_knot = parameterize.initial_guess_curve(curve, eval_point)

    assert eval_knot == test_knot


@pytest.mark.parametrize('knot_val, expected',
                     [
                         (0.25, 0.25),
                         (0.5, 0.5),
                         (0.75, 0.75),
                     ]
                     )
def test_curve_inversion(curve, knot_val, expected):
    eval_point = curve.single_point(knot_val)
    inverted_knot = parameterize.curveinversion(curve, eval_point)

    # atol condition selection is adhoc
    condition = np.isclose(knot_val, inverted_knot, atol=1e-2)

    assert condition
