"""

    Tests for Spline module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from .context import np
from pysplinefit import Spline

import pytest


@pytest.fixture
def curve():
    """ Generate curve for text """
    curve = Spline.Curve()
    curve.degree = 3
    curve.control_points = np.array([[0, 0],
                                     [1, 2],
                                     [2, 2],
                                     [3, 0]])
    curve.knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    return curve


def test_curve_degree(curve):
    assert curve.degree == 3


def test_curve_control_points(curve):
    given_pts = np.array([[0, 0],
                          [1, 2],
                          [2, 2],
                          [3, 0]])

    condition = np.allclose(given_pts, curve.control_points)

    assert condition


def test_curve_knot_vector(curve):
    give_knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    condition = np.allclose(give_knot_vector, curve.knot_vector)

    assert condition


def test_curve_control_point_guard():
    with pytest.raises(Exception):
        given_pts = np.array([[0, 0],
                              [1, 2],
                              [2, 2],
                              [3, 0]])

        curve = Spline.Curve()

        curve.control_points = given_pts


def test_curve_knot_vector_guard():
    with pytest.raises(Exception):
        give_knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        curve = Spline.Curve()

        curve.knot_vector = give_knot_vector


def test_curve_knot_vector_guard2():
    with pytest.raises(Exception):
        give_knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        curve = Spline.Curve()

        curve.degree = 3

        curve.knot_vector = give_knot_vector
