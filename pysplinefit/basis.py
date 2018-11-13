"""
.. module:: basis
    :platform: Unix, Windows
    :synopsis: Basis functions for NURBS and B-Spline Curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def basis_functions(knot_span, knot, degree, knot_vector):

    """
    Algorithm A2.2 from Piegl & Tiller, The NURBS Book, 1997

    Compute non-vanishing basis functions for a given knot value and knot index

    :param knot_span: knot vector span containing knot
    :type knot_span: int
    :param knot: value of knot
    :type knot: float
    :param degree: degree
    :type: int
    :param knot_vector: knot vector containing knot and knot span
    :type knot_vector: ndarray, list, tuple
    :return: Array containing values of non-vanishing basis functions evaluated at knot
    :rtype: ndarray
    """

    # Initialize empty array to hold the degree + 1 non-vanishing basis values. Note N[0] = 1.0 by def
    N = np.ones(degree + 1)

    # Initialize empty array to hold left and right computation values
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)

    # Account for the fact that range goes up to max - 1
    for j in range(1, degree + 1):
        # Setup left and right values
        left[j] = knot - knot_vector[knot_span + 1 - j]
        right[j] = knot_vector[knot_span + j] - knot
        saved = 0.0

        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N
