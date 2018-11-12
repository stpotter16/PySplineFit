"""
.. module:: basis
    :platform: Unix, Windows
    :synopsis: Basis functions for NURBS and B-Spline Curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def find_span(num_ctrlpts, degree, knot, knot_vector, **kwargs):

    """
    Algorithm A2.1 from Piegl & Tiller, The NURBS Book, 1997

    Find the knot span index in knot vector for a given knot value

    :param num_ctrlpts: number of control points
    :type num_ctrlpts: int
    :param degree: degree
    :type: int
    :param knot: value of knot
    :type knot: float
    :param knot_vector: knot vector to search in
    :type knot_vector: ndarray, list, tuple
    :return: index of knot interval containing knot
    :rtype: int
    """

    # Number of knot intervals, n, are based on number of control points.
    # Per convention in The NURBS Book, num_ctrlpts = n + 1
    n = num_ctrlpts - 1

    # Edge case: Return highest knot interval (n) if the knot is equal to the knot vector value in that span
    # Extract relative tolerance
    rtol = kwargs.get('rtol', 1e-6)

    # Compare
    if np.allclose(knot, knot_vector[n + 1], rtol=rtol):
        return n

    # Begin binary search
    # Set low and high
    low = degree
    high = num_ctrlpts

    #
