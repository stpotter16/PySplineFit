"""
.. module:: spline
    :platform: Unix, Windows
    :synopsis: Contains high level classes for containing NURBS and B-Spline Curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import knots
from . import basis
from . import fileIO


class Curve:
    """
    Class for Spline curves
    """

    def __init__(self):
        self._degree = None
        self._control_points = None
        self._knot_vector = None

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def degree(self):
        """
        Degree of spline curve

        :getter: Gets spline curve degree
        :type: int
        """
        return self._degree

    @degree.setter
    def degree(self, deg):
        if deg <= 0:
            raise ValueError('Degree must be greater than or equal to one')
        if not isinstance(deg, int):
            try:
                deg = int(deg)  # Cast degree to int
            except Exception:
                print('Input value for degree was of invalid type and is unable to be cast to an int')
                raise

        self._degree = deg

    @property
    def control_points(self):
        """
        Control points of spline curve

        :getter: Gets spline curve control points
        :type: ndarray
        """
        return self._control_points

    @control_points.setter
    def control_points(self, ctrlpt_array):

        # Check that degree has been set
        if self._degree is None:
            raise ValueError('Curve degree must be set before setting control points')

        # Check that input is okay
        if not isinstance(ctrlpt_array, np.ndarray):
            try:
                ctrlpt_array = np.array(ctrlpt_array)  # Try to cast to an array
            except Exception:
                print('Input value for control points was of invalid type and is unable to be cast to an array')
                raise

        # Check that the shape and size is correct
        if ctrlpt_array.ndim != 2:
            raise ValueError('Control point array must be 2D')

        # Check that the control points are at either 2D or 3D
        if not (ctrlpt_array.shape[-1] == 2 or ctrlpt_array.shape[-1] == 3):
            raise ValueError('Control point points must be in either R2 or R3')

        self._control_points = ctrlpt_array

    @property
    def knot_vector(self):
        """
        Knot vector of spline curve

        :getter: Gets spline curve knot vector
        :type: ndarray
        """
        return self._knot_vector

    @knot_vector.setter
    def knot_vector(self, knot_vector_array):

        # Check that degree has been set
        if self._degree is None:
            raise ValueError('Curve degree must be set before setting knot vector')

        # Check that control points are set
        if self._control_points is None:
            raise ValueError("Curve control points must be set before setting knot vector")

        # Check that input is okay
        if not isinstance(knot_vector_array, np.ndarray):
            try:
                knot_vector_array = np.array(knot_vector_array)  # Cast to array
            except Exception:
                print('Input value for knot vector was of invalid type and is unable to be cast to an array')
                raise

        # Check that knot vector is valid
        if not knots.check_knot_vector(self._degree, knot_vector_array, len(self._control_points)):
            raise ValueError('Knot vector is invalid')

        self._knot_vector = knot_vector_array

    def single_point(self, knot):
        """
        Evaluate a curve at a single parametric point

        :param knot: parameter at which to evaluate the curve
        :type knot: float
        :return: value of curve at that parameteric location as an array [X1, X2, X3]
        :rtype: ndarray
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
        knot_span = knots.find_span(len(self._control_points), self._degree, knot, self._knot_vector)

        # Evaluate basis functions
        basis_funs = basis.basis_functions(knot_span, knot, self._degree, self._knot_vector)

        # Pull out active control points
        active_control_points = self._control_points[knot_span - self._degree:knot_span + 1, :]

        point = np.array([basis_funs @ active_control_points[:, 0],
                          basis_funs @ active_control_points[:, 1],
                          basis_funs @ active_control_points[:, 2]])

        return point

    def points(self, knot_array):
        """
        Evaluate the curve at multiple parameters

        :param knot_array: array of parameter values
        :type knot_array: ndarray
        :return: array of evaluated curve points
        :rtype: ndarray
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
        """
        Evaluate derivatives of the curve at specified knot up to min(point, degree)

        :param knot: point to evaluate
        :type knot: ndarray
        :param order: max order of derivative
        :type order: int
        :param normalize: Optional. Boolean switch to control normalization of output derivatives
        :type normalize: bool
        :return: Array of points and derivatives at specified knot
        :rtype: ndarray
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
        knot_span = knots.find_span(len(self._control_points), self._degree, knot, self._knot_vector)

        # Basis function derivatives
        basis_fun_ders = basis.basis_function_ders(knot_span, knot, self._degree, self.knot_vector, max_order)

        # Pull out active control points
        active_control_points = self._control_points[knot_span - self._degree:knot_span + 1, :]

        # Compute point and derivatives
        derivs = np.zeros((max_order + 1, 3))

        for row in range(len(derivs)):
            val = np.array([basis_fun_ders[:, row] @ active_control_points[:, 0],
                            basis_fun_ders[:, row] @ active_control_points[:, 1],
                            basis_fun_ders[:, row] @ active_control_points[:, 2]])
            if normalize and not np.isclose(np.linalg.norm(val), 0.0):
                val = val / np.linalg.norm(val)

            derivs[row, :] = val

        return derivs

    def insert_knot(self, knot_val):

        """
        Insert single knot of specified knot value into the knot vector of the curve

        :param knot_val: knot value to be inserted
        :type knot_val: float
        :return: None
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
        new_knot_vector, new_control_points = knots.curve_knot_insertion(self._degree, self._knot_vector,
                                                                         self._control_points, knot_val)

        # Set these new values
        self.control_points = new_control_points
        self.knot_vector = new_knot_vector

    def save(self, name='curve.txt'):
        """
        Save curve object to file

        :param name: Optional. Path (relative or absolute) to file to save to. Default 'curve.txt'
        :type name: str
        :return: None
        :rtype: None
        """

        # Check input
        if not isinstance(name, str):
            try:
                name = str(name)
            except Exception:
                print('Input file name was not a string type and could not be cast')
                raise

        # Call fileIO function
        fileIO.write_curve_to_txt(self, name)

    def load(self, name):
        """
        Load curve object from file and set degree, control points, and knot vector

        :param name: Path (relative or absolute) to file to load from
        :type name: str
        :return: None
        :rtype: None
        """

        # Check input
        if not isinstance(name, str):
            try:
                name = str(name)
            except Exception:
                print('Input file name was not a string and could not be cast')
                raise

        # Call fileIO function
        fileIO.read_curve_from_txt(self, name)


class Surface:
    """
    Class for Spline surfaces
    """

    def __init__(self):
        self._degree_u = None
        self._degree_v = None
        self._num_ctrlpts_u = None
        self._num_ctrlpts_v = None
        self._control_points = None
        self._knot_vector_u = None
        self._knot_vector_v = None

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def degree_u(self):
        """
        Degree of spline curve

        :getter: Gets spline curve degree_u
        :type: int
        """
        return self._degree_u

    @degree_u.setter
    def degree_u(self, deg):
        if deg <= 0:
            raise ValueError('Degree must be greater than or equal to one')
        if not isinstance(deg, int):
            try:
                deg = int(deg)  # Cast degree_u to int
            except Exception:
                print('Input value for degree_u was of invalid type and is unable to be cast to an int')
                raise

        self._degree_u = deg

    @property
    def degree_v(self):
        """
        Degree of spline curve

        :getter: Gets spline curve degree_v
        :type: int
        """
        return self._degree_v

    @degree_v.setter
    def degree_v(self, deg):
        if deg <= 0:
            raise ValueError('Degree must be greater than or equal to one')
        if not isinstance(deg, int):
            try:
                deg = int(deg)  # Cast degree_v to int
            except Exception:
                print('Input value for degree_v was of invalid type and is unable to be cast to an int')
                raise

        self._degree_v = deg

    @property
    def num_ctrlpts_u(self):
        """
        Number of surface control points in the u direction

        :getter: Get spline surface control point count in u
        :type: int
        """
        return self._num_ctrlpts_u

    @num_ctrlpts_u.setter
    def num_ctrlpts_u(self, num):
        # Sanitize input
        if not isinstance(num, int):
            try:
                num = int(num)
            except Exception:
                print('Number of control points in u was not an int and could not be cast')
                raise Exception

        if num <= 3:
            raise ValueError('Number of control points in u must be 3 or more')

        self._num_ctrlpts_u = num

    @property
    def num_ctrlpts_v(self):
        """
        Number of surface control points in the u direction

        :getter: Get spline surface control point count in u
        :type: int
        """
        return self._num_ctrlpts_v

    @num_ctrlpts_v.setter
    def num_ctrlpts_v(self, num):
        # Sanitize input
        if not isinstance(num, int):
            try:
                num = int(num)
            except Exception:
                print('Number of control points in v was not an int and could not be cast')
                raise Exception

        if num <= 3:
            raise ValueError('Number of control points in v must be 3 or more')

        self._num_ctrlpts_v = num

    @property
    def control_points(self):
        """
        Control points of spline surface

        ordered by stepping through v then u

        :getter: Gets spline surface control points
        :type: ndarray
        """
        return self._control_points

    @control_points.setter
    def control_points(self, ctrlpt_array):

        # Check that degrees has been set
        if self._degree_u is None:
            raise ValueError('Surface degree u must be set before setting control points')

        if self._degree_v is None:
            raise ValueError('Surface degree v must be set before setting control points')

        # Check that input is okay
        if not isinstance(ctrlpt_array, np.ndarray):
            try:
                ctrlpt_array = np.array(ctrlpt_array)  # Try to cast to an array
            except Exception:
                print('Input value for control points was of invalid type and is unable to be cast to an array')
                raise

        # Check that the shape and size is correct
        if ctrlpt_array.ndim != 2:
            raise ValueError('Control point array must be 2D')

        # Check that the control points are at either 2D or 3D
        if not (ctrlpt_array.shape[-1] == 2 or ctrlpt_array.shape[-1] == 3):
            raise ValueError('Control point points must be in either R2 or R3')

        self._control_points = ctrlpt_array
