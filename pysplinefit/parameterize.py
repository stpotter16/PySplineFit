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


def curve_inversion(curve, point, eps1=1e-5, eps2=1e-5, max_iter=100):

    """
    Uses Newton-Rapshon iteration to find the parametric value u [0, 1] that corresponds to the closest point on a 3D
    Spline curve to a 3D data point.

    Based on algorithms laid out in Section 6.1 in Piegl & Tiller's "The NURBS Book"

    :param curve: 3D Spline curve onto which the data point is being projected.
    :type curve: spline.Curve()object
    :param point: Point in 3D space that is being projected onto the curve
    :type point: numpy array of floats
    :param eps1: termination tolerance condition for Euclidean distance. Optional. Default 1e-5
    :type eps1: float
    :param eps2: termination tolerance condition for zero cosine. Optional. Default 1e-5
    :type eps2: float
    :param max_iter: Optional. maximum number of iterations for the Newton-Raphson algorithm. Default 100
    :type max_iter: int
    :return: value of the curve parameter u [0, 1] corresponding to closest point on the Spine curve.
    :rtype: float
    """
    # Extract initial parameter guess.

    u0 = initial_guess_curve(curve, point)
    ui = u0
    ders_i = curve.derivatives(ui, 2, normalize=False)
    curve_pt_i = ders_i[0, :]
    curve_u_i = ders_i[1, :]
    curve_uu_i = ders_i[2, :]

    # First convergence check at ui. Check to see if distance between point and curve is less than tolerance. P&T (6.4)
    cond1 = np.linalg.norm(curve.single_point(ui) - point)
    if cond1 <= eps1:
        return ui

    # Second convergence check at ui. Check to see if the angle between the point and the curve is orthogonal. P&T (6.4)
    cond2 = np.linalg.norm(np.inner(curve_u_i, (curve_pt_i - point))) / (np.linalg.norm(curve_u_i)
                                                                         * np.linalg.norm((curve_pt_i - point)))
    if cond2 <= eps2:
        return ui

    # If these checks don't terminate, iterate with Newton Raphson
    j = 0
    while True:
        # Calculate new parameter according to P&T (6.3)
        num = np.inner(curve_u_i, (curve_pt_i - point))
        denom = np.inner(curve_uu_i, (curve_pt_i - point)) + np.linalg.norm(curve_u_i)**2
        uip1 = ui - num/denom

        # Make sure u stays in bounds (Third condition). Adjustments change based on whether or not curve is closed.
        if uip1 < 0.0:
            uip1 = 0.0
        elif uip1 > 1.0:
            uip1 = 1.0

        # Fourth convergence check at ui
        val = [i * (uip1 - ui) for i in curve_u_i]  # Scalar multiplication for list. curve u i not numpy array
        cond4 = np.linalg.norm(val)
        if cond4 <= eps1:
            break

        # If the last two convergence checks don't terminate, calculate info at new parameter (uip1)
        ders_ip1 = curve.derivatives(uip1, 2, normalize=False)
        curve_pt_ip1 = ders_ip1[0, :]
        curve_u_ip1 = ders_ip1[1, :]
        curve_uu_ip1 = ders_ip1[2, :]

        # First convergence check at uip1. P&T (6.4)
        cond1 = np.linalg.norm(curve.single_point(uip1) - point)
        if cond1 <= eps1:
            break

        # Second convergence check at uip1. P&T (6.4)
        cond2 = np.linalg.norm(np.inner(curve_u_ip1, (curve_pt_ip1 - point)))/(np.linalg.norm(curve_u_ip1)
                                                                               * np.linalg.norm((curve_pt_ip1 - point)))
        if cond2 <= eps2:
            break

        # Pass values forward to next iteration.
        ui = uip1
        curve_pt_i = curve_pt_ip1
        curve_u_i = curve_u_ip1
        curve_uu_i = curve_uu_ip1

        # If max iteration value hit, terminate.
        if j > max_iter:
            break
        j += 1

    # Return curve parameter value.
    return ui


def parameterize_curve(curve, data):
    """
    Parameterize given data points to give curve with curve inversion method

    :param curve: Curve to parameterize data to
    :type curve: spline.Curve() object
    :param data: Data to parameterize
    :type data: ndarray
    :return: Data point array with parametrization value appended [X1, X2, X3, u]
    :rtype: ndarray
    """
    # Parameterize
    knot_vals = [curve_inversion(curve, point) for point in data]

    # Cast to numpy array
    knot_vals = np.array(knot_vals)

    # Append knot vals to data
    return np.column_stack((data, knot_vals))
