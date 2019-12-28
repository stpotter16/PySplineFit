"""
.. module:: base_geometry
    :platform: Unix, Windows
    :synopsis: Base classes for geometry
"""

import numpy as np

from spline.algorithms.basis import basis_functions, basis_function_ders
from spline.algorithms.knots import curve_knot_insertion, check_knot_vector find_span
from spline.model.base_geometry import SplineCurve, SplineSurface


class BSplineCurve(SplineCurve):
    """ A B-Spline curve
    """

    def single_point(self, knot):
        """ Evaluate a curve at a single parametric point

        Parameters
        ----------
        knot : float
            Parameter at which to evaluate curve

        Returns
        -------
        point : array
            Evalued coordinate point 
        """
        # Check that value is is a float
        if not isinstance(knot, float):
            try:
                knot = float(knot)  # Try to cast
            except Exception:
                print('Parameter must be a float. Could not cast input to float')
                raise

        # Make sure value is range [0, 1]
        if not (0 <= knot <= 1):
            raise ValueError('Parameter must be in the interval [0, 1]')

        # Get knot span
        knot_span = find_span(len(self._control_points), self._degree, knot, self._knot_vector)

        # Evaluate basis functions
        basis_funs = basis_functions(knot_span, knot, self._degree, self._knot_vector)

        # Pull out active control points
        active_control_points = self._control_points[knot_span - self._degree:knot_span + 1, :]

        point = np.array([basis_funs @ active_control_points[:, 0],
                          basis_funs @ active_control_points[:, 1],
                          basis_funs @ active_control_points[:, 2]])

        return point

    def points(self, knot_array):
        """ Evaluate the curve at multiple parametric locations

        Parameters
        ----------
        knots : array
            Array of parametric points (u) to evaluate
        
        Returns
        -------
        points : array
            Evaluated coordinat points
        """
        # Check input
        if not isinstance(knot_array, np.ndarray):
            try:
                knot_array = np.array(knot_array)  # Try type conversion
            except Exception:
                print('Input parameters was not an array type and could not be cast to an array')
                raise

        # Make sure input is one dimensional
        if knot_array.ndim != 1.0:
            raise ValueError('Parameter array must be 1D')

        values = [self.single_point(parameter) for parameter in knot_array]

        return np.array(values)

    def derivatives(self, knot, order, normalize=True):
        """ Evaluate the derivatives of the curve at specified knot up to
        min (order, degree)

        Parameters
        ----------
        knot : float
            Parametric point to evaluate
        order : int
            Max order of derivatives to evaluate
        normalize : bool, optional
            Normalize output derivatives
        
        Returns
        -------
        derivs : array
            Array of points and derivatives at specified knot
        """
        # Check inputs
        if not isinstance(knot, float):
            try:
                knot = float(knot)
            except Exception:
                print('Knot input must be a float and could not be cast')
                raise

        if not isinstance(order, int):
            try:
                order = int(order)
            except Exception:
                print('Derivative order input must be a int and could not be cast')
                raise

        # Check knot
        if not (0.0 <= knot <= 1.0):
            raise ValueError('Knot must be in interval [0, 1]')

        # Check derivative
        if order <= 1:
            raise ValueError('Derivative order must be greater than zero')

        # Set maximum number of derivatives
        max_order = min(order, self._degree)

        # Find span
        knot_span = find_span(len(self._control_points), self._degree, knot, self._knot_vector)

        # Basis function derivatives
        basis_fun_ders = basis_function_ders(knot_span, knot, self._degree, self.knot_vector, max_order)

        # Pull out active control points
        active_control_points = self._control_points[knot_span - self._degree:knot_span + 1, :]

        # Compute point and derivatives
        derivs = np.zeros((max_order + 1, 3))

        for row in range(len(derivs)):
            val = np.array([basis_fun_ders[:, row] @ active_control_points[:, 1],
                            basis_fun_ders[:, row] @ active_control_points[:, 2]])
            if normalize and not np.isclose(np.linalg.norm(val), 0.0) and row != 0:
                val = val / np.linalg.norm(val)

            derivs[row, :] = val

        return derivs

    def insert_knot(self, knot_val):
        """ Insert a single knot of specified value into the curves knot vector

        Parameters
        ----------
        knot: float
            Knot value to be inserted
        """
        # Check input
        if not isinstance(knot_val, float):
            try:
                knot_val = float(knot_val)
            except Exception:
                print('Input knot value is not a float type and was unable to be cast to a float')
                raise

        if not (0 < knot_val < 1):
            raise ValueError('Knot value must be in the interval [0, 1]')

        # Compute new knot vectors and control points
        new_knot_vector, new_control_points = curve_knot_insertion(self._degree, self._knot_vector,
                                                                         self._control_points, knot_val)

        # Set these new values
        self.control_points = new_control_points
        self.knot_vector = new_knot_vector

    def _check_knot_vector(self, kv):
        """ Check that knot vector is valid
        """
        return check_knot_vector(self._degree, kv, len(self._control_points))


class BSplineSurface(SplineSurface):
    """ A B-Spline Surface
    """
    def single_point(self, u, v):
        """ Evaluate a surface at a single parametric point

        Parameters
        ----------
        knot_u : float
            Parameter in u at which to evaluate surface 
        knot_v : float
            Parameter in v at withch to evaluate surface

        Returns
        -------
        point : array
            Evalued coordinate point 
        """
        # Check values
        if not isinstance(u, float):
            try:
                u = float(u)
            except Exception:
                print('u value needs to be a float and was unable to cast')
                raise
        if not isinstance(v, float):
            try:
                v = float(v)
            except Exception:
                print('v value needs to be a float and was unable to cast')
                raise

        # Make sure valuges are in interval [0, 1]
        if not (0 <= u <= 1):
            raise ValueError('u parameter must be in interval [0, 1]')
        if not (0 <= v <= 1):
            raise ValueError('v parameter must be in interval [0, 1]')

        # Get knot spans
        u_span = find_span(self._num_ctrlpts_u, self._degree_u, u, self._knot_vector_u)
        v_span = find_span(self._num_ctrlpts_v, self._degree_v, v, self._knot_vector_v)

        # Evaluate basis functions
        basis_funs_u = basis_functions(u_span, u, self._degree_u, self._knot_vector_u)
        basis_funs_v = basis_functions(v_span, v, self._degree_v, self._knot_vector_v)

        # Create the matrix of control point values
        ctrlpt_x = self._control_points[:, 0]
        ctrlpt_y = self._control_points[:, 1]
        ctrlpt_z = self._control_points[:, 2]

        x_array = np.reshape(ctrlpt_x, (self._num_ctrlpts_u, self._num_ctrlpts_v))
        y_array = np.reshape(ctrlpt_y, (self._num_ctrlpts_u, self._num_ctrlpts_v))
        z_array = np.reshape(ctrlpt_z, (self._num_ctrlpts_u, self._num_ctrlpts_v))

        x = basis_funs_u @ x_array[u_span - self._degree_u:u_span + 1, v_span - self._degree_v:v_span + 1] \
            @ basis_funs_v
        y = basis_funs_u @ y_array[u_span - self._degree_u:u_span + 1, v_span - self._degree_v:v_span + 1] \
            @ basis_funs_v
        z = basis_funs_u @ z_array[u_span - self._degree_u:u_span + 1, v_span - self._degree_v:v_span + 1] \
            @ basis_funs_v

        point = np.array([x, y, z])

        return point

    def points(self, knot_array):
        """ Evaluate the surface at multiple parametric locations

        Parameters
        ----------
        knots : array
            Array of parametric points (u,v) to evaluate
        
        Returns
        -------
        points : array
            Evaluated coordinat points
        """

        # Check input
        if not isinstance(knot_array, np.ndarray):
            try:
                knot_array = np.array(knot_array)
            except Exception:
                print('Input parameter array was not an array type anc could not be cast to an array')
                raise

        # Make sure input is two dimensional
        if knot_array.ndim != 2.0:
            raise ValueError('Parameter array must be 2D')

        values = [self.single_point(parameter[0], parameter[1]) for parameter in knot_array]

        return np.array(values)

    def derivatives(self, u, v, order_u, order_v, normalize=True):
        """ Evaluate the derivatives of the surface at specified knot up to
        min (order, degree)

        Parameters
        ----------
        knot_u : float
            Parametric in u point to evaluate
        knot_v : float
            Parametric in u point to evaluate
        order_u : int
            Max order of derivatives in u to evaluate
        order_v : int
            Max order of derivatives in v to evaluate
        normalize : bool, optional
            Normalize output derivatives
        
        Returns
        -------
        derivs : array
            Array of points and derivatives at specified knot
        """
 
        # Check inputs
        # Check values
        if not isinstance(u, float):
            try:
                u = float(u)
            except Exception:
                print('u value needs to be a float and was unable to cast')
                raise
        if not isinstance(v, float):
            try:
                v = float(v)
            except Exception:
                print('v value needs to be a float and was unable to cast')
                raise

        # Make sure valuges are in interval [0, 1]
        if not (0 <= u <= 1):
            raise ValueError('u parameter must be in interval [0, 1]')
        if not (0 <= v <= 1):
            raise ValueError('v parameter must be in interval [0, 1]')

        # Set max derivative orders
        max_order_u = min(order_u, self._degree_u)
        max_order_v = min(order_v, self._degree_v)

        # Get knot spans
        u_span = find_span(self._num_ctrlpts_u, self._degree_u, u, self._knot_vector_u)
        v_span = find_span(self._num_ctrlpts_v, self._degree_v, v, self._knot_vector_v)

        # Evaluate basis functions
        basis_funs_u_ders = basis_function_ders(u_span, u, self._degree_u, self._knot_vector_u, max_order_u)
        basis_funs_v_ders = basis_function_ders(v_span, v, self._degree_v, self._knot_vector_v, max_order_v)

        # Create the matrix of control point values
        ctrlpt_x = self._control_points[:, 0]
        ctrlpt_y = self._control_points[:, 1]
        ctrlpt_z = self._control_points[:, 2]

        x_array = np.reshape(ctrlpt_x, (self._num_ctrlpts_u, self._num_ctrlpts_v))
        y_array = np.reshape(ctrlpt_y, (self._num_ctrlpts_u, self._num_ctrlpts_v))
        z_array = np.reshape(ctrlpt_z, (self._num_ctrlpts_u, self._num_ctrlpts_v))

        # Active control point
        x_active = x_array[u_span - self._degree_u:u_span + 1, v_span - self._degree_v:v_span + 1]
        y_active = y_array[u_span - self._degree_u:u_span + 1, v_span - self._degree_v:v_span + 1]
        z_active = z_array[u_span - self._degree_u:u_span + 1, v_span - self._degree_v:v_span + 1]

        # Compute derivatives
        derivs = np.zeros(((max_order_u + 1) * (max_order_v + 1), 3))

        # Loop through and fill derivatives array
        index = 0
        for u_row in range(0, max_order_u + 1):
            for v_row in range(0, max_order_v + 1):

                # Compute x, y, z components
                x = basis_funs_u_ders[:, u_row] @ x_active @ basis_funs_v_ders[:, v_row]
                y = basis_funs_u_ders[:, u_row] @ y_active @ basis_funs_v_ders[:, v_row]
                z = basis_funs_u_ders[:, u_row] @ z_active @ basis_funs_v_ders[:, v_row]

                val = np.array([x, y, z])

                if normalize and not np.isclose(np.linalg.norm(val), 0.0) and index != 0:
                    val = val / np.linalg.norm(val)

                derivs[index, :] = val

                index += 1

        return derivs

    def _check_knot_vector(self, kv, direction='u'):
        """ Check that knot vector is valid
        """
        pass
