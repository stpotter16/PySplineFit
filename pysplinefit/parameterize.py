"""
.. module:: parameterize
    :platform: Unix, Windows
    :synopsis: Contains functions for parameterizing point data to nurbs and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def initial_guess_curve(curve, point):
    """
    Finds the location of the closest span of a spline curve to a 3D data point.

    Output is used as the initial guess for a Newton-Raphson iterative minimization algorithm.

    :param curve: 3D curve onto which to project the data point
    :type curve: splint.Curve() object
    :param point: Point in 3D space that is being projected onto the curve
    :type point: ndarray
    :return: value of the curve parameter u in the closest span [0, 1]
    :rtype: float
    """

    # Set array of values of parameter u to evaluate
    num_u_spans = 2 * len(curve.control_points)
    eval_u = 1 / (num_u_spans - 1) * np.array(list(range(0, num_u_spans)))
    u0 = eval_u[0]

    # Set minimum value as a hugh number to start with
    min_val = 10.0 ** 6

    # Loop through list of evaluation knots
    for u in range(0, num_u_spans):
        r = point - curve.single_point(eval_u[u])
        normr = np.linalg.norm(r, 2)
        if normr < min_val:
            min_val = normr
            u0 = eval_u[u]

    return u0
