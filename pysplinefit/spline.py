"""
.. module:: spline
    :platform: Unix, Windows
    :synopsis: Contains high level classes for containing NURBS and B-Spline Curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import knots
from . import basis


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
