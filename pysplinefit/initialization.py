"""
.. module:: initialization
    :platform: Unix, Windows
    :synopsis: Contains functions for initializing spline curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import spline
from . import knots


def initialize_curve(start, end, degree, num_ctrlpts):

    """
    Create and return an initial curve with a fixed number of control points and specified degree between the left and
    right points

    :param start: Start point for initial curve. A point in 3D space
    :type start: ndarray
    :param end: End point for initial curve. A point in 3D space
    :type end: ndarray
    :param degree: Degree of initial curve.
    :type degree: int
    :param num_ctrlpts: Number of control points in initial curve. Must be degree + 1
    :type num_ctrlpts: int
    :return: Initial curve as an instance of the spline.Curve() class
    :rtype: spline.Curve()
    """

    # Check inputs
    if not num_ctrlpts >= degree:
        raise ValueError('Number of control points must be greater than or equal to the degree')

    # Initialize curve
    curve = spline.Curve()

    # Set degree
    curve.degree = degree

    # Set control points
    x_vals = np.linspace(start[0], end[0], num_ctrlpts)
    y_vals = np.linspace(start[1], end[1], num_ctrlpts)
    z_vals = np.linspace(start[2], end[2], num_ctrlpts)

    init_ctrlpts = np.column_stack((x_vals, y_vals, z_vals))

    curve.control_points = init_ctrlpts

    # Generate knot vector
    init_knot_vector = knots.generate_uniform(degree, len(init_ctrlpts))

    curve.knot_vector = init_knot_vector

    return curve
