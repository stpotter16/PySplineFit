"""
.. module:: fitting
    :platform: Unix, Windows
    :synopsis: Contains functions for fitting point data to curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import basis


def linear_fit_fixed(R, Q, P, fixed, logging=1):
    """
    Linear least squares fitting with fixed values

    :param R: Observation matrix
    :type R: ndarray
    :param Q: right hand side vector
    :type Q: ndarray
    :param P: left hand vector
    :type P: ndarray
    :param fixed: Boolean array corresponding to length of P. If P[i] is true, i value is fixed
    :type fixed: ndarray
    :param logging: Option switch for log level. <=0 silent, =1 normal, >1 debug
    :type logging: int
    :return: Least squares solutions
    :rtype: ndarray
    """

    # Set fixed points
    nfixed = 0
    for i in range(0, len(P)):
        if fixed[i]:
            nfixed += 1
    if nfixed == len(P):
        return P

    # set free matrix components
    Rfree = np.zeros((len(Q), len(P) - nfixed))
    Rfixed = np.zeros((len(Q), nfixed))
    Qfixed = np.zeros(len(Q))
    j = k = 0
    for i in range(0, len(P)):
        if fixed[i]:
            Rfixed[:, j] = R[:, i]
            Qfixed += R[:, i] * P[i]
            j += 1
        else:
            Rfree[:, k] = R[:, i]
            k += 1

    # Perform the least squares
    if logging >= 2:
        a_free = np.matmul(Rfree.T, Rfree)
        print('Condition number of free RTR')
        print(np.linalg.cond(a_free))
    freeP = np.linalg.lstsq(Rfree, (Q - Qfixed), rcond=None)[0]
    fullP = np.zeros(len(P))
    j = 0
    for i in range(0, len(P)):
        if fixed[i]:
            fullP[i] = P[i]
        else:
            fullP[i] = freeP[j]
            j += 1
    return fullP


def single_fit(curve, parameterized_data, logging=1):
    """
    Perform single fit of curve to data

    :param curve: Spline curve to fit
    :type curve: spline.Curve() object
    :param parameterized_data: Data to fit the curve to
    :type parameterized_data: ndarray
    :param logging: Option switch for log level. <=0 silent, =1 normal, >1 debug
    :type logging: int
    :return: Fit curve
    :rtype: spline.Curve() object
    """
    # Set up evaluation data
    eval_u = parameterized_data[:, -1]
    ctrl_u = len(curve.control_points)

    n_mat = np.zeros((len(parameterized_data), len(curve.control_points)))
    n_i = basis.one_basis_function

    # Fill matrix
    row = 0
    col = 0

    if logging >= 1:
        print('Filling R Matrix')

    for u in range(0, len(eval_u)):
        for spanu in range(0, ctrl_u):
            n_mat[row, col] = n_i(curve.degree, curve.knot_vector, spanu, eval_u[u])
            col += 1
        row += 1
        col = 0

    # Check partition of unity
    if logging >= 1:
        print('Checking partition of unity')
    unity = np.sum(n_mat, axis=1)
    if logging >= 2:
        print(unity)
    total = np.sum(unity)
    if not np.isclose(total, len(parameterized_data)):
        raise ValueError('Basis does not conform to partition of unity')

    # Prepare matrices for fitting
    n_trans = np.transpose(n_mat)
    a_mat = np.matmul(n_trans, n_mat)

    # Check conditioning
    condition = np.linalg.cond(a_mat)
    if logging >= 1:
        print('Conditioning number of RTR: {}'.format(condition))

    # Setup fixed point array
    fixed = np.zeros(len(curve.control_points), dtype=bool)
    fixed[0], fixed[-1] = True, True

    # Fit via least squares
    if logging >= 1:
        print('Performing least squares')

    ctrlpt_x = linear_fit_fixed(n_mat, parameterized_data[:, 0], curve.control_points[:, 0], fixed)
    ctrlpt_y = linear_fit_fixed(n_mat, parameterized_data[:, 1], curve.control_points[:, 1], fixed)
    ctrlpt_z = linear_fit_fixed(n_mat, parameterized_data[:, 2], curve.control_points[:, 2], fixed)

    # Update control points
    new_control_point = np.column_stack((ctrlpt_x, ctrlpt_y, ctrlpt_z))

    curve.control_points = new_control_point
