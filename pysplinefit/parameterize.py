"""
.. module:: parameterize
    :platform: Unix, Windows
    :synopsis: Contains functions for parameterizing point data to curves and surfaces

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
    min_val = 1e6

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

    Sort results in ascending parameter order

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
    param_data = np.column_stack((data, knot_vals))

    param_data = np.sort(param_data.view('float64,float64,float64,float64'), order=['f3'],
                         axis=0).view(np.float)
    # Above from stack exchange https://stackoverflow.com/q/2828059

    return param_data


def initial_guess_surf(surface, point):

    """
    Finds the location of the closest span (u, v) of a spline surface to a 3D data point.

    Output is used as the initial guess for a Newton-Raphson iterative minimization algorithm.

    :param surface: 3D Spline surface onto which the data point is being projected.
    :type surface: spline.Surface object
    :param point: Point in 3D space that is being projected onto the surf
    :type point: numpy array of floats
    :return: value of the surface parameter (u, v) in the closest span [0, 1] x [0, 1] as an array
    :rtype: ndarray
    """
    # Set array of values of (u, v) to evaluate
    num_u_spans = 2 * surface.num_ctrlpts_u
    num_v_spans = 2 * surface.num_ctrlpts_v
    eval_u = 1 / (num_u_spans - 1) * np.array(list(range(0, num_u_spans)))
    eval_v = 1 / (num_v_spans - 1) * np.array(list(range(0, num_v_spans)))
    u0 = eval_u[0]
    v0 = eval_v[0]

    # Set minimum value
    min_val = 1e6

    # Evaluate surface. Careful, this assumes the surface is open in both directions.
    for u in range(0, num_u_spans):
        for v in range(0, num_v_spans):
            r = point - surface.single_point(eval_u[u], eval_v[v])
            normr = np.linalg.norm(r, 2)
            if normr < min_val:
                min_val = normr
                u0 = eval_u[u]
                v0 = eval_v[v]

    return np.array([u0, v0])


def surfinversion(num_u_spans, num_v_spans, surf, point, eps1, eps2, max_iter=100, u_closed=False, v_closed=False):

    """
    Uses Newton-Rapshon iteration to find the parametric coordinate (u,v) [0, 1] x [0, 1] that corresponds to the closest
    point on a 3D NURBS surface to a 3D data point.

    Based on algorithms laid out in Section 6.1 in Piegl & Tiller's "The NURBS Book"

    ROOM FOR IMPROVEMENT

    :param num_u_spans: number of intervals in the u knot vector to evaluate to find minimum distance span.
    :type num_u_spans: int
    :param num_v_spans: number of intervals in the v knot vector to evaluate to find minimum distance
    :type num_v_spans: int
    :param surf: 3D NURBS surface onto which the data point is being projected.
    :type surf: NURBS surface object
    :param point: Point in 3D space that is being projected onto the curve
    :type point: numpy array of floats
    :param eps1: termination tolerance condition for Euclidean distance.
    :type eps1: float
    :param eps2: termination tolerance condition for zero cosine.
    :type eps2: float
    :param max_iter: maximum number of iterations for the Newton-Raphson algorithm
    :type max_iter: int
    :param u_closed: optional. Boolean to indicate whether or not surface is closed in u direction
    :type u_closed: bool
    :param v_closed: optional. Boolean to indicate whether or not surface is closed in v direction
    :type v_closed: bool
    :return: value of the surface parametric coordinate (u,v) [0, 1] x [0, 1] corresponding to closest point on the
    NURBS surface.
    :rtype: numpy array
    """
    # Extract initial parameter guess.
    guess = initialguesssurf(num_u_spans, num_v_spans, surf, point)
    u0, v0 = guess[0], guess[1]
    ui = u0
    vi = v0
    ders_i = surf.derivatives(ui, vi, 2)
    surf_u_i = np.array(ders_i[1][0])
    surf_v_i = np.array(ders_i[0][1])
    surf_uu_i = np.array(ders_i[2][0])
    surf_vv_i = np.array(ders_i[0][2])
    surf_uv_i = np.array(ders_i[1][1])
    surf_vu_i = np.array(ders_i[1][1])

    # First convergence check at (ui, vi). Check if distance between point and surface is less than tolerance. P&T (6.8)
    cond1 = np.linalg.norm(surf.evaluate_single((ui, vi)) - point)
    if cond1 <= eps1:
        return np.array([ui, vi])

    # Second convergence check at (ui, vi). Check if angle between point and surface is orthogonal. P&T (6.8)
    cond2a = np.linalg.norm(np.inner(surf_u_i, (surf.evaluate_single((ui, vi)) - point))) / (
                np.linalg.norm(surf_u_i) * np.linalg.norm(surf.evaluate_single((ui, vi)) - point))
    cond2b = np.linalg.norm(np.inner(surf_v_i, (surf.evaluate_single((ui, vi)) - point))) / (
                np.linalg.norm(surf_v_i) * np.linalg.norm(surf.evaluate_single((ui, vi)) - point))

    if cond2a <= eps2 and cond2b <= eps2:
        return np.array([ui, vi])

    # If these checks don't terminate, iterate with Newton Raphson
    n = 0
    while True:
        # Calculate new parameter
        r = surf.evaluate_single((ui, vi)) - point
        f = np.inner(r, surf_u_i)
        g = np.inner(r, surf_v_i)
        fu = np.linalg.norm(surf_u_i)**2 + np.inner(r, surf_uu_i)
        fv = np.inner(surf_u_i, surf_v_i) + np.inner(r, surf_uv_i)
        gu = np.inner(surf_u_i, surf_v_i) + np.inner(r, surf_vu_i)
        gv = np.linalg.norm(surf_v_i)**2 + np.inner(r, surf_vv_i)
        # P&T (6.5)
        jacobian = np.array([[fu, fv],
                             [gu, gv]])
        kappa = -1.0 * np.array([[f],
                                 [g]])
        # P&T (6.6)
        delta = np.matmul(np.linalg.inv(jacobian), kappa)

        # Keep parameter in range (third condition)
        if (delta[0, 0] + ui) < 0.0:
            if u_closed:
                uip1 = 1 - (0 - (delta[0, 0] + ui))
                if uip1 < 0.0:
                    uip1 = ui  # SO BRITTLE
            else:
                uip1 = 0.0
        elif (delta[0, 0] + ui) > 1.0:
            if u_closed:
                uip1 = 0 + ((delta[0, 0] + ui) - 1)
                if uip1 > 1.0:
                    uip1 = ui  # SO BRITTLE
            else:
                uip1 = 1.0
        else:
            uip1 = delta[0, 0] + ui

        if (delta[1, 0] + vi) < 0.0:
            if v_closed:
                vip1 = 1 - (0 - (delta[1, 0] + vi))
                if vip1 < 0.0:
                    vip1 = vi  # SO BRITTLE
            else:
                vip1 = 0.0
        elif (delta[1, 0] + vi) > 1.0:
            if v_closed:
                vip1 = 0 + ((delta[1, 0] + vi) - 1)
                if vip1 > 1.0:
                    vip1 = vi  # SO BRITTLE
            else:
                vip1 = 1.0
        else:
            vip1 = delta[1, 0] + vi

        # Fourth convergence check at (ui, vi)
        cond4 = np.linalg.norm((uip1 - ui)*surf_u_i + (vip1 - vi)*surf_v_i)
        if cond4 <= eps1:
            break

        # Calculate info at new parameters (uip1, vip1)
        ders_ip1 = surf.derivatives(uip1, vip1, 2)
        surf_u_ip1 = np.array(ders_ip1[1][0])
        surf_v_ip1 = np.array(ders_ip1[0][1])
        surf_uu_ip1 = np.array(ders_ip1[2][0])
        surf_vv_ip1 = np.array(ders_ip1[0][2])
        surf_uv_ip1 = np.array(ders_ip1[1][1])
        surf_vu_ip1 = np.array(ders_ip1[1][1])

        # First convergence check at (uip1, vip1)
        cond1 = np.linalg.norm(surf.evaluate_single((uip1, vip1)) - point)
        if cond1 <= eps1:
            break

        # Second convergence check at (uip1, vip1)
        cond2a = np.linalg.norm(np.inner(surf_u_ip1,
                                         (surf.evaluate_single((uip1, vip1))
                                          - point)))/(np.linalg.norm(surf_u_ip1)
                                                      * np.linalg.norm(surf.evaluate_single((uip1, vip1)) - point))
        cond2b = np.linalg.norm(np.inner(surf_v_ip1,
                                         (surf.evaluate_single((uip1, vip1))
                                          - point)))/(np.linalg.norm(surf_v_ip1)
                                                      * np.linalg.norm(surf.evaluate_single((uip1, vip1)) - point))

        if cond2a <= eps2 and cond2b <= eps2:
            break

        # Check maximum iterations
        if n >= max_iter:
            break

        n += 1

        # Pass values forward to next iteration
        ui = uip1
        vi = vip1
        surf_u_i = surf_u_ip1
        surf_v_i = surf_v_ip1
        surf_uu_i = surf_uu_ip1
        surf_vv_i = surf_vv_ip1
        surf_uv_i = surf_uv_ip1
        surf_vu_i = surf_vu_ip1

    return np.array([uip1, vip1])
