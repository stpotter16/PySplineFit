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
from pysplinefit import knots

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
    inverted_knot = parameterize.curve_inversion(curve, eval_point)

    # atol condition selection is adhoc. Should set more carefully
    condition = np.isclose(knot_val, inverted_knot, atol=1e-2)

    assert condition


def test_parameterize_curve(curve):
    test_knots = np.linspace(0.1, 0.9, 10)
    test_points = curve.points(test_knots)

    parameterized_test_points = parameterize.parameterize_curve(curve, test_points)

    condition = np.allclose(test_knots, parameterized_test_points[:, -1], atol=1e-3)

    assert condition


@pytest.fixture
def surf():
    """ Generate surface for testing """
    surf = spline.Surface()
    surf.degree_u = 2
    surf.degree_v = 2
    surf.num_ctrlpts_u = 5
    surf.num_ctrlpts_v = 5

    x = np.arange(0.0, 5.0)
    y = np.arange(0.0, 5.0)

    ys, xs = np.meshgrid(x, y)

    theta = np.linspace(0, 2 * np.pi, len(xs.flatten()))
    z = np.sin(theta)

    ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), z))

    surf.control_points = ctrlpt_array

    uvec = knots.generate_uniform(surf.degree_u, surf.num_ctrlpts_u)
    vvec = knots.generate_uniform(surf.degree_v, surf.num_ctrlpts_v)

    surf.knot_vector_u = uvec
    surf.knot_vector_v = vvec

    return surf


def test_initial_guess_surf(surf):
    eval_u = 1 / (2 * surf.num_ctrlpts_u - 1) * 2
    eval_v = 1 / (2 * surf.num_ctrlpts_v - 1) * 3
    eval_point = surf.single_point(eval_u, eval_v)

    test_val = parameterize.initial_guess_surf(surf, eval_point)

    condition = np.allclose(np.array([eval_u, eval_v]), test_val)

    assert condition
