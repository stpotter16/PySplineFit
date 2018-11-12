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

    # Compute midpoint sum
    mid_sum = low + high

    # Case structure if mid_sum is odd or even
    # If even, mid value is straight forward average
    if mid_sum % 2 == 0:
        mid = mid_sum / 2
    # If odd, add 1 to mid_sum to make even, then divide
    else:
        mid = (mid_sum + 1) / 2

    # Cast result as int so it works as an idex
    mid = int(mid)

    # While loop structure for binary search
    while knot < knot_vector[mid] or knot > knot_vector[mid + 1]:
        # Update high/low value
        if knot < knot_vector[mid]:
            high = mid
        else:
            low = mid
        # Update mid value
        mid = int((low + high) / 2)

    return mid
