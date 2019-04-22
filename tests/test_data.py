"""

    Tests for data module of the PySplineFit Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2019 Sam Potter

    Requires pytest
"""

from .context import pysplinefit
from .context import np
from pysplinefit import data

import pytest


@pytest.fixture
def bound():
    """ Generate boundary for testing"""
    bound = data.Boundary()
    bound.degree = 3
    bound.start = np.array([0.0, 0.0, 0.0])
    bound.end = np.array([1.0, 0.0, 0.0])
    bound.num_ctrlpts = 5

    return bound

def test_boundary_degree(bound):
    assert bound.degree == 3


def test_boundary_start(bound):
    condition = np.allclose(bound.start, np.array([0.0, 0.0, 0.0]))
    assert condition


def test_boundary_end(bound):
    condition = np.allclose(bound.end, np.array([1.0, 0.0, 0.0]))
    assert condition


def test_boundary_num_ctrlpts(bound):
    assert bound.num_ctrlpts == 5


@pytest.fixture
def interior():
    """ Generate interior for testing"""
    interior = data.Interior()
    interior.degree = 3
    interior.num_ctrlpts = 5

    return interior


def test_interior_degree(interior):
    assert interior.degree == 3


def test_interior_num_ctrlpts(interior):
    assert interior.num_ctrlpts == 5
