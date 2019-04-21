"""
.. module:: spline
    :platform: Unix, Windows
    :synopsis: Contains high level classes for containing NURBS and B-Spline Curves and surfaces

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import knots


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
    def degree(self, value):
        if value <= 0:
            raise ValueError('Degree must be greater than or equal to one')
        if not isinstance(value, int):
            try:
                value = int(value)  # Cast degree to int
            except Exception:
                print('Input value for degree was of invalid type and is unable to be cast to an int')
                raise

        self._degree = value

    @property
    def control_points(self):
        """
        Control points of spline curve

        :getter: Gets spline curve control points
        :type: ndarray
        """
        return  self._control_points

    @control_points.setter
    def control_points(self, array):

        # Check that degree has been set
        if self._degree is None:
            raise ValueError('Curve degree must be set before setting control points')

        # Check that input is okay
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array) # Try to cast to an array
            except Exception:
                print('Input value for control points was of invalid type and is unable to be cast to an array')
                raise

        # Check that the shape and size is correct
        if array.ndim != 2:
            raise ValueError('Control point array must be 2D')

        # Check that the control points are at either 2D or 3D
        if not (array.shape[-1] == 2 or array.shape[-1] == 3):
            raise ValueError('Control point points must be in either R2 or R3')

        self._control_points = array

    @property
    def knot_vector(self):
        """
        Knot vector of spline curve

        :getter: Gets spline curve knot vector
        :type: ndarray
        """
        return self._knot_vector

    @knot_vector.setter
    def knot_vector(self, array):
        
        # Check that degree has been set
        if self._degree is None:
            raise ValueError('Curve degree must be set before setting knot vector')

        # Check that control points are set
        if self._control_points is None:
            raise ValueError("Curve control points must be set before setting knot vector")

        # Check that input is okay
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)  # Cast to array
            except Exception:
                print('Input value for knot vector was of invalid type and is unable to be cast to an array')
                raise

        # Check that knot vector is valid
        if not knots.check_knot_vector(self._degree, array, len(self._control_points)):
            raise ValueError('Knot vector is invalid')

        self._knot_vector = array
