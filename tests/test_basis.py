"""

    Tests for basis module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from pysplinefit import basis

import pytest
import numpy as np


def test_basis_functions():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    knot_span = 4
    knot = 5.0/2.0

    # The NURBS Book Ex. 2.3
    basis_vals = basis.basis_functions(knot_span, knot, degree, knot_vector)

    expected = np.array([0.125, 0.75, 0.125])

    condition = np.allclose(basis_vals, expected)

    assert condition


def test_basis_functions2():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    knot_span = 4
    knot = 5.0/2.0

    # The NURBS Book Ex. 2.3
    basis_vals = basis.basis_functions(knot_span, knot, degree, knot_vector)

    basis_sum = np.sum(basis_vals)

    assert np.isclose(basis_sum, 1.0)


def test_basis_function_ders():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    knot_span = 4
    knot = 5.0/2.0
    deriv_order = 2

    # The NURBS Book Ex. 2.4
    ders_vals = basis.basis_function_ders(knot_span, knot, degree, knot_vector, deriv_order)

    expected = np.array([[0.125, -0.5, 1.0],
                         [0.75, 0, -2.0],
                         [0.125, 0.5, 1.0]])

    condition = np.allclose(ders_vals, expected)

    assert condition


def test_one_basis_function():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    knot = 5.0/2.0

    # The NURBS Book Ex. 2.5
    basis_val1 = basis.one_basis_function(degree, knot_vector, 3, knot)
    basis_val2 = basis.one_basis_function(degree, knot_vector, 4, knot)

    basis_vals = np.array([basis_val1, basis_val2])

    expected = np.array([0.75, 0.125])

    condition = np.allclose(basis_vals, expected)

    assert condition


def test_one_basis_function_ders():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1
    knot_span = 4
    knot = 5.0/2.0
    deriv_order = 2

    # The NURBS Book Ex. 2.4
    basis_deriv_vals = basis.one_basis_function_ders(degree, knot_vector, knot_span, knot, deriv_order)

    expected = np.array([0.125, 0.5, 1.0])

    condition = np.allclose(basis_deriv_vals, expected)

    assert condition
