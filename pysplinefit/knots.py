"""
.. module:: knots
    :platform: Unix, Windows
    :synopsis: Functions for handling knot vectors for NURBS and B-Spline Curves and surfaces

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
    # Edge case: Return highest knot interval (n) if the knot is equal to the knot vector value in that span
    # Extract relative tolerance
    rtol = kwargs.get('rtol', 1e-6)

    # Compare
    if np.allclose(knot, knot_vector[num_ctrlpts], rtol=rtol):
        return num_ctrlpts - 1

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


def normalize(knot_vector):
    """
    Normalize input knot vector to [0, 1].


    :param knot_vector: Knot vector to be normalized
    :type knot_vector: ndarray
    :return: Normalized knot vector
    :rtype: ndarray
    """

    # Confirm input is numpy array
    if not isinstance(knot_vector, np.ndarray):
        try:
            np.array(knot_vector)
        except Exception:
            print('Knot vector input not a numpy array and could not convert')
            raise

    # Sanitize input
    if knot_vector.ndim != 1:
        raise ValueError('Knot vector must be a 1D array')

    max_knot = np.max(knot_vector)
    min_knot = np.min(knot_vector)

    knot_vector = knot_vector - min_knot * np.ones(knot_vector.shape)
    knot_vector *= 1 / (max_knot - min_knot)

    return knot_vector


def check_knot_vector(degree, knot_vector, num_ctrlpts):
    """
    Confirm that the knot vector conforms the the rules for B Splines

    :param degree: degree of spline basis
    :type degree: int
    :param knot_vector: knot vector
    :type knot_vector: ndarray, list, tuple
    :param num_ctrlpts: number of control points associated with knot vector
    :type num_ctrlpts: int
    :return: Bool on whether or not knot vector conforms. True is conforming, False nonconforming
    :rtype: bool
    """

    # Confirm input is numpy array
    if not isinstance(knot_vector, np.ndarray):
        try:
            knot_vector = np.array(knot_vector)
        except Exception:
            print('Knot vector input not a numpy array and could not convert')
            raise

    # Sanitize input
    if knot_vector.ndim != 1:
        raise ValueError('Knot vector must be a 1D array')

    # Normalize knot vector
    knot_vector = normalize(knot_vector)

    # Check that the length is correct
    if len(knot_vector) != num_ctrlpts + degree + 1:
        return False

    # Check that the first degree + 1 values are zero
    if not np.allclose(np.zeros(degree + 1), knot_vector[:degree + 1]):
        return False

    # Check that the last degree + 1 values are 1
    if not np.allclose(np.ones(degree + 1), knot_vector[-1 * (degree + 1)]):
        return False

    # Check that the knots are increasing
    previous_knot = knot_vector[0]
    for knot in knot_vector:
        if knot < previous_knot:
            return False
        previous_knot = knot

    return True


def generate_uniform(degree, num_ctrlpts):
    """
    Generates uniform, clamped knot vector on [0, 1] given basis degree and number of control points.

    :param degree: degree of basis
    :type degree: int
    :param num_ctrlpts: number of control points
    :type num_ctrlpts: int
    :return: Uniform, clamped knot vector on [0, 1] as an array
    :rtype: ndarray
    """

    # specify length of total knot vector
    length_knot_vector = num_ctrlpts + degree + 1

    # Subract off the repeated knots and beginning and end
    num_middle_knots = length_knot_vector - 2 * degree

    # Create evenly spaced knots from 0 to 1 of number num_middle_knots
    middle_knot_vector = np.linspace(0, 1, num_middle_knots)

    # Append middle knot vector with repeated knots on beginning and end
    knot_vector = np.concatenate((np.zeros(degree), middle_knot_vector, np.ones(degree)))

    return knot_vector


def find_multiplicity(knot, knot_vector):

    """
    Helper function for finding the multiplicity of a given knot in a given knot vector

    :param knot: knot
    :type knot: float
    :param knot_vector: knot vector to search in
    :type knot_vector: ndarray, list, tuple
    :return: multiplicity of knot
    :rtype: int
    """

    mult = 0

    for knot_span in knot_vector:
        if np.isclose(knot, knot_span):
            mult += 1

    return mult


def curve_knot_insertion(degree, old_knot_vector, old_ctrlpts, inserted_knot, num_inserts=1):
    """
    Algorithm A5.1, The NURBS Book, 1997

    Inserts knot found in knot span of old knot vector with given multiplicity a given number of times and returns the
    new knot vector

    Values for knot span and multiplicity are found via helper functions

    :param degree: degree
    :type degree: int
    :param old_knot_vector: original knot vector
    :type old_knot_vector: ndarray, list, tuple
    :param old_ctrlpts: original weighted control points
    :type old_ctrlpts: ndarray, list, tuple
    :param inserted_knot: knot to be inserted
    :type inserted_knot: float
    :param num_inserts: number of times to insert knot. Default 1
    :type num_inserts: int
    :return: Tuple of (new knot vector, new weighted control points)
    :rtype: tuple
    """

    # Find span and multiplicity
    inserted_knot_span = find_span(len(old_ctrlpts), degree, inserted_knot, old_knot_vector)

    knot_multiplicity = find_multiplicity(inserted_knot, old_knot_vector)

    # Knot vector lengths
    old_knot_vector_length = len(old_ctrlpts) + degree + 1
    new_knot_vector_length = len(old_knot_vector) + num_inserts

    # Create new knot vector and control point array
    new_knot_vector = np.zeros(new_knot_vector_length)
    new_ctrlpts = np.zeros((len(old_ctrlpts) + num_inserts, 3))
    R = np.zeros((degree + 1, 3))

    # Load new values
    for i in range(0, inserted_knot_span + 1):
        new_knot_vector[i] = old_knot_vector[i]

    for i in range(1, num_inserts + 1):
        new_knot_vector[inserted_knot_span + i] = inserted_knot

    for i in range(inserted_knot_span + 1, old_knot_vector_length):
        new_knot_vector[i + num_inserts] = old_knot_vector[i]

    # Save unaltered control points
    for i in range(0, inserted_knot_span - degree + 1):
        new_ctrlpts[i, :] = old_ctrlpts[i, :]

    for i in range(inserted_knot_span - knot_multiplicity, len(old_ctrlpts)):
        new_ctrlpts[i + num_inserts] = old_ctrlpts[i, :]

    for i in range(0, degree - knot_multiplicity + 1):
        R[i, :] = old_ctrlpts[inserted_knot_span - degree + i, :]

    # Insert knot
    for j in range(1, num_inserts + 1):
        L = inserted_knot_span - degree + j

        for i in range(0, degree - j - knot_multiplicity + 1):
            alpha = (inserted_knot - old_knot_vector[L + i]) / (old_knot_vector[i + inserted_knot_span + 1]
                                                                - old_knot_vector[L + i])
            R[i, :] = alpha * R[i + 1, :] + (1.0 - alpha) * R[i, :]

        new_ctrlpts[L, :] = R[0, :]
        new_ctrlpts[inserted_knot_span + num_inserts - j - knot_multiplicity, :] = R[degree - j - knot_multiplicity, :]

    # Load new the rest of the control points
    L = inserted_knot_span - degree + num_inserts
    for i in range(L + 1, inserted_knot_span - knot_multiplicity):
        new_ctrlpts[i, :] = R[i - L, :]

    return (new_knot_vector, new_ctrlpts)
