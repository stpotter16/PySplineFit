"""

    Tests for Spline module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from .context import np
from pysplinefit import spline
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
def test_curve_single_point(curve2, knot_val, expected):
    evalpt = curve2.single_point(knot_val)

    assert np.isclose(evalpt[0], expected[0])
    assert np.isclose(evalpt[1], expected[1])
    assert np.isclose(evalpt[2], expected[2])


def test_curve_points(curve2):
    inputs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.array([[0.0, 0.0, 0.0],
                         [1.22222222, 2.44444444, 0.0],
                         [2.5, 4.0, 0.0],
                         [3.77777778, 2.44444444, 0.0],
                         [5.0, 0.0, 0.0]])

    points = curve2.points(inputs)

    condition = np.allclose(points, expected)

    assert condition


@pytest.mark.parametrize('knot_val, expected',
                         [
                             (0.0, (0.0, 0.0, 0.0)),
                             (0.25, (1.2222222222, 2.444444444444, 0.0)),
                             (0.5, (2.5, 4.0, 0.0)),
                             (0.75, (3.77777777778, 2.44444444444, 0.0)),
                             (1.0, (5.0, 0.0, 0.0))
                         ]
                         )
def test_curve_derivpt(curve2, knot_val, expected):
    deriv_pt = curve2.derivatives(knot_val, 2, normalize=False)[0, :]

    assert np.isclose(deriv_pt[0], expected[0])
    assert np.isclose(deriv_pt[1], expected[1])
    assert np.isclose(deriv_pt[2], expected[2])


@pytest.mark.parametrize('knot_val, expected',
                         [
                             (0.0, (1.0, 2.0, 0.0)),
                             (0.25, (1.0, 2.0, 0.0)),
                             (0.75, (1.0, -2.0, 0.0)),
                             (1.0, (1.0, -2.0, 0.0))
                         ]
                         )
def test_curve_deriv1(curve2, knot_val, expected):
    deriv1 = curve2.derivatives(knot_val, 2)[1, :]

    expected = np.array(expected)

    expected = expected / np.linalg.norm(expected)

    assert np.allclose(deriv1, expected)


def test_curve_insertion(curve2):
    new_knot = 0.875

    eval_knot = 0.7

    old_point = curve2.single_point(eval_knot)

    curve2.insert_knot(new_knot)

    new_point = curve2.single_point(eval_knot)

    condition = np.allclose(old_point, new_point)

    assert condition


def test_curve_insertion2(curve2):

    new_knot = 0.875

    new_knot_vector = np.array([0, 0, 0, 0.375, 0.5, 0.625, 0.875, 1, 1, 1])

    curve2.insert_knot(new_knot)

    condition = np.allclose(new_knot_vector, curve2.knot_vector)

    assert condition


def test_curve_insertion3(curve2):
    new_knot = 0.875

    old_num_ctrlpts = len(curve2.control_points)

    curve2.insert_knot(new_knot)

    assert old_num_ctrlpts == len(curve2.control_points) - 1


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

    ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

    surf.control_points = ctrlpt_array

    uvec = knots.generate_uniform(surf.degree_u, surf.num_ctrlpts_u)
    vvec = knots.generate_uniform(surf.degree_v, surf.num_ctrlpts_v)

    surf.knot_vector_u = uvec
    surf.knot_vector_v = vvec

    return surf


def test_surf_deg_u(surf):

    assert surf.degree_u == 2


def test_surf_deg_v(surf):

    assert surf.degree_v == 2


def test_surf_num_ctrlpts_u(surf):

    assert surf.num_ctrlpts_u == 5


def test_surf_num_ctrlpts_v(surf):

    assert surf.num_ctrlpts_v == 5


def test_surf_ctrlpts(surf):
    x = np.arange(0.0, 5.0)
    y = np.arange(0.0, 5.0)

    ys, xs = np.meshgrid(x, y)

    ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

    condition = np.allclose(ctrlpt_array, surf.control_points)

    assert condition


def test_knot_vector_u(surf):

    uvec = knots.generate_uniform(surf.degree_u, surf.num_ctrlpts_u)

    condition = np.allclose(uvec, surf.knot_vector_u)

    assert condition


def test_knot_vector_v(surf):

    vvec = knots.generate_uniform(surf.degree_v, surf.num_ctrlpts_v)

    condition = np.allclose(vvec, surf.knot_vector_v)

    assert condition


def test_surf_control_pt_guard_u():
    with pytest.raises(Exception):
        x = np.arange(0.0, 5.0)
        y = np.arange(0.0, 5.0)

        ys, xs = np.meshgrid(x, y)

        ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

        surf = spline.Surface()
        surf.degree_u = 2

        surf.control_points = ctrlpt_array


def test_surf_control_pt_guard_v():
    with pytest.raises(Exception):
        x = np.arange(0.0, 5.0)
        y = np.arange(0.0, 5.0)

        ys, xs = np.meshgrid(x, y)

        ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

        surf = spline.Surface()
        surf.degree_v = 2

        surf.control_points = ctrlpt_array


def test_surf_knot_vector_guard_u():
    with pytest.raises(Exception):
        x = np.arange(0.0, 5.0)
        y = np.arange(0.0, 5.0)

        ys, xs = np.meshgrid(x, y)

        ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

        surf = spline.Surface()
        surf.degree_u = 2
        surf.degree_v = 2

        num_ctrlpts = 5

        knot_vec = knots.generate_uniform(surf.degree_u, num_ctrlpts)

        surf.knot_vector_u = knot_vec


def test_surf_knot_vector_guard_u2():
    with pytest.raises(Exception):
        x = np.arange(0.0, 5.0)
        y = np.arange(0.0, 5.0)

        ys, xs = np.meshgrid(x, y)

        ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

        surf = spline.Surface()
        surf.degree_u = 2
        surf.degree_v = 2

        num_ctrlpts = 5

        surf.control_points = ctrlpt_array

        knot_vec = knots.generate_uniform(surf.degree_u, num_ctrlpts)

        surf.knot_vector_u = knot_vec


def test_surf_knot_vector_guard_v():
    with pytest.raises(Exception):
        x = np.arange(0.0, 5.0)
        y = np.arange(0.0, 5.0)

        ys, xs = np.meshgrid(x, y)

        ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

        surf = spline.Surface()
        surf.degree_u = 2
        surf.degree_v = 2

        num_ctrlpts = 5

        knot_vec = knots.generate_uniform(surf.degree_v, num_ctrlpts)

        surf.knot_vector_v = knot_vec


def test_surf_knot_vector_guard_v2():
    with pytest.raises(Exception):
        x = np.arange(0.0, 5.0)
        y = np.arange(0.0, 5.0)

        ys, xs = np.meshgrid(x, y)

        ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

        surf = spline.Surface()
        surf.degree_u = 2
        surf.degree_v = 2

        num_ctrlpts = 5

        surf.control_points = ctrlpt_array

        knot_vec = knots.generate_uniform(surf.degree_v, num_ctrlpts)

        surf.knot_vector_v = knot_vec


@pytest.fixture
def surf2():
    """ Generate surface for testing """
    surf = spline.Surface()
    surf.degree_u = 2
    surf.degree_v = 2
    surf.num_ctrlpts_u = 5
    surf.num_ctrlpts_v = 5

    x = np.arange(0.0, 5.0)
    y = np.arange(0.0, 5.0)

    ys, xs = np.meshgrid(x, y)

    ctrlpt_array = np.column_stack((xs.flatten(), ys.flatten(), np.zeros(len(xs.flatten()))))

    surf.control_points = ctrlpt_array

    uvec = knots.generate_uniform(surf.degree_u, surf.num_ctrlpts_u)
    vvec = knots.generate_uniform(surf.degree_v, surf.num_ctrlpts_v)

    surf.knot_vector_u = uvec
    surf.knot_vector_v = vvec

    return surf


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (0.0, 0.0, 0.0)),
                             (0.25, 0.25, (1.21875, 1.21875, 0.0)),
                             (0.75, 0.75,  (2.78125, 2.78125, 0.0)),
                             (0.5, 0.25, (2.0, 1.21875, 0.0)),
                             (0.25, 0.5, (1.21875, 2.0, 0.0)),
                             (1.0, 1.0,  (4.0, 4.0, 0.0))
                         ]
                         )
def test_surf_single_point(surf2, u_val, v_val, expected):
    eval = surf2.single_point(u_val, v_val)

    expected = np.array(expected)

    condition = np.allclose(eval, expected)

    assert condition


def test_surf_points(surf2):
    knot_vals = np.array([[0.0, 0.0],
                          [0.25, 0.25],
                          [0.75, 0.75],
                          [0.5, 0.25],
                          [0.25, 0.5],
                          [1.0, 1.0]
                          ])

    evalpts = surf2.points(knot_vals)

    expected = np.array([[0.0, 0.0, 0.0],
                         [1.21875, 1.21875, 0.0],
                         [2.78125, 2.78125, 0.0],
                         [2.0, 1.21785, 0.0],
                         [1.21875, 2.0, 0.0],
                         [4.0, 4.0, 0.0]
                         ])

    # Adhoc selection of atol again
    condition = np.allclose(evalpts, expected, atol=1e-3)

    assert condition


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (0.0, 0.0, 0.0)),
                             (0.25, 0.25, (1.21875, 1.21875, 0.0)),
                             (0.75, 0.75,  (2.78125, 2.78125, 0.0)),
                             (0.5, 0.25, (2.0, 1.21875, 0.0)),
                             (0.25, 0.5, (1.21875, 2.0, 0.0)),
                             (1.0, 1.0,  (4.0, 4.0, 0.0))
                         ]
                         )
def test_surf_derivpt(surf2, u_val, v_val, expected):
    eval = surf2.derivatives(u_val, v_val, 2, 2)[0, :]

    expected = np.array(expected)

    condition = np.allclose(eval, expected)

    assert condition


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (6.0, 0.0, 0.0)),
                             (0.25, 0.25, (3.75, 0.0, 0.0)),
                             (0.75, 0.75,  (3.75, 0.0, 0.0)),
                             (0.5, 0.25, (3.0, 0.0, 0.0)),
                             (0.25, 0.5, (3.75, 0.0, 0.0)),
                             (1.0, 1.0,  (6.0, 0.0, 0.0))
                         ]
                         )
def test_surf_derivu1(surf2, u_val, v_val, expected):
    eval = surf2.derivatives(u_val, v_val, 2, 2, normalize=False)[3, :]

    expected = np.array(expected)

    # Add hoc atol
    condition = np.allclose(eval, expected, atol=1e-3)

    assert condition


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (-9.0, 0.0, 0.0)),
                             (0.25, 0.25, (-9.0, 0.0, 0.0)),
                             (0.75, 0.75,  (9.0, 0.0, 0.0)),
                             (0.5, 0.25, (0.0, 0.0, 0.0)),
                             (0.25, 0.5, (-9.0, 0.0, 0.0)),
                             (1.0, 1.0,  (9.0, 0.0, 0.0))
                         ]
                         )
def test_surf_derivu2(surf2, u_val, v_val, expected):
    eval = surf2.derivatives(u_val, v_val, 2, 2, normalize=False)[6, :]

    expected = np.array(expected)

    # Add hoc atol
    condition = np.allclose(eval, expected, atol=1e-3)

    assert condition


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (0.0, 6.0, 0.0)),
                             (0.25, 0.25, (0.0, 3.75, 0.0)),
                             (0.75, 0.75,  (0.0, 3.75, 0.0)),
                             (0.5, 0.25, (0.0, 3.75, 0.0)),
                             (0.25, 0.5, (0.0, 3.0, 0.0)),
                             (1.0, 1.0,  (0.0, 6.0, 0.0))
                         ]
                         )
def test_surf_derivv1(surf2, u_val, v_val, expected):
    eval = surf2.derivatives(u_val, v_val, 2, 2, normalize=False)[1, :]

    expected = np.array(expected)

    # Add hoc atol
    condition = np.allclose(eval, expected, atol=1e-3)

    assert condition


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (0.0, -9.0, 0.0)),
                             (0.25, 0.25, (0.0, -9.0, 0.0)),
                             (0.75, 0.75,  (0.0, 9.0, 0.0)),
                             (0.5, 0.25, (0.0, -9.0, 0.0)),
                             (0.25, 0.5, (0.0, 0.0, 0.0)),
                             (1.0, 1.0,  (0.0, 9.0, 0.0))
                         ]
                         )
def test_surf_derivv2(surf2, u_val, v_val, expected):
    eval = surf2.derivatives(u_val, v_val, 2, 2, normalize=False)[2, :]

    expected = np.array(expected)

    # Add hoc atol
    condition = np.allclose(eval, expected, atol=1e-3)

    assert condition


@pytest.mark.parametrize('u_val, v_val, expected',
                         [
                             (0.0, 0.0,  (0.0, 0.0, 0.0)),
                             (0.25, 0.25, (0.0, 0.0, 0.0)),
                             (0.75, 0.75,  (0.0, 0.0, 0.0)),
                             (0.5, 0.25, (0.0, 0.0, 0.0)),
                             (0.25, 0.5, (0.0, 0.0, 0.0)),
                             (1.0, 1.0,  (0.0, 0.0, 0.0))
                         ]
                         )
def test_surf_derivuv(surf2, u_val, v_val, expected):
    eval = surf2.derivatives(u_val, v_val, 2, 2, normalize=False)[4, :]

    expected = np.array(expected)

    # Add hoc atol
    condition = np.allclose(eval, expected, atol=1e-3)

    assert condition
