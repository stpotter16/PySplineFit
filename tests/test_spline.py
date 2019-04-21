"""

    Tests for Spline module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from .context import np
from pysplinefit import spline

import pytest


@pytest.fixture
def curve():
    """ Generate curve for text """
    curve = spline.Curve()
    curve.degree = 3
    curve.control_points = np.array([[0, 0, 0],
                                     [1, 2, 0],
                                     [2, 2, 0],
                                     [3, 0, 0]])
    curve.knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    return curve


def test_curve_degree(curve):
    assert curve.degree == 3


def test_curve_control_points(curve):
    given_pts = np.array([[0, 0, 0],
                          [1, 2, 0],
                          [2, 2, 0],
                          [3, 0, 0]])

    condition = np.allclose(given_pts, curve.control_points)

    assert condition


def test_curve_knot_vector(curve):
    give_knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    condition = np.allclose(give_knot_vector, curve.knot_vector)

    assert condition


def test_curve_control_point_guard():
    with pytest.raises(Exception):
        given_pts = np.array([[0, 0, 0],
                              [1, 2, 0],
                              [2, 2, 0],
                              [3, 0, 0]])

        curve = spline.Curve()

        curve.control_points = given_pts


def test_curve_knot_vector_guard():
    with pytest.raises(Exception):
        give_knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        curve = spline.Curve()

        curve.knot_vector = give_knot_vector


def test_curve_knot_vector_guard2():
    with pytest.raises(Exception):
        give_knot_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        curve = spline.Curve()

        curve.degree = 3

        curve.knot_vector = give_knot_vector


@pytest.fixture
def curve2():
    """ Generate another curve for testing purposes """
    curve2 = spline.Curve()
    curve2.degree = 2
    curve2.control_points = np.array([[0.0, 0.0, 0.0],
                                      [1.0, 2.0, 0.0],
                                      [2.0, 4.0, 0.0],
                                      [3.0, 4.0, 0.0],
                                      [4.0, 2.0, 0.0],
                                      [5.0, 0.0, 0.0]])
    curve2.knot_vector = np.array([0, 0, 0, 0.375, 0.5, 0.625, 1, 1, 1])

    return curve2


@pytest.mark.parametrize('knot_val, expected',
                         [
                             (0.0, (0.0, 0.0, 0.0)),
                             (0.25, (1.2222222222, 2.444444444444, 0.0)),
                             (0.5, (2.5, 4.0, 0.0)),
                             (0.75, (3.77777777778, 2.44444444444, 0.0)),
                             (1.0, (5.0, 0.0, 0.0))
                         ]
                         )
def test_curve_point(curve2, knot_val, expected):
    evalpt = curve2.single_point(knot_val)

    assert np.isclose(evalpt[0], expected[0])
    assert np.isclose(evalpt[1], expected[1])
    assert np.isclose(evalpt[2], expected[2])