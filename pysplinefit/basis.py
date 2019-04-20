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


def basis_function_ders(knot_span, knot, degree, knot_vector, deriv_order):
    """
    Algorithm A2.3 from Piegl & Tiller, The NURBS Book, 1997

    Compute non-vanishing basis functions and associated derivatives up to a specified order for a given knot value and
    knot index

    :param knot_span: knot vector span containing knot
    :type knot_span: int
    :param knot: value of knot
    :type knot: float
    :param degree: degree
    :type: int
    :param knot_vector: knot vector containing knot and knot span
    :type knot_vector: ndarray, list, tuple
    :param deriv_order: highest order of derivative to be computed. deriv_order <= degree
    :type deriv_order: int
    :return: Array containing values of non-vanishing basis functions and all derivative orders up to deriv_order
    evaluated at knot
    :rtype: ndarray
    """

    # Initialize output and local arrays
    ders = np.zeros((degree + 1, deriv_order + 1))
    # ders[basis function# (knot_span - degree + row #), derivative order]
    # Note, this deviates from the structure found in the NURBS book
    ndu = np.zeros((degree + 1, degree + 1))
    ndu[0, 0] = 1.0
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    a = np.zeros((2, degree + 1))

    # Create basis function triangles
    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[knot_span + 1 - j]
        right[j] = knot_vector[knot_span + j] - knot
        saved = 0.0

        for r in range(0, j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]

            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        ndu[j, j] = saved

    # Fill in basis function values (no derivative)
    for j in range(0, degree + 1):
        ders[j, 0] = ndu[j, degree]

    # Compute derivatives
    for r in range(0, degree + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0

        # Loop to kth derivative
        for k in range(1, deriv_order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2, 0] = a[s2, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = degree - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]

            ders[r, k] = d

            # Swap rows of a
            j = s1
            s1 = s2
            s2 = j

    # Multiply correction factors
    r = degree
    for k in range(1, deriv_order + 1):
        for j in range(0, degree + 1):
            ders[j, k] *= r
            r *= (degree - k)

    return ders
