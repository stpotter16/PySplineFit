"""
.. module:: data
    :platform: Unix, Windows
    :synopsis: Contains high level classes for containing point cloud boundary and surface data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import spline
from . import np


class Boundary(spline.Curve):
    """
    Class container for boundary data
    """

    def __init__(self):
        super().__init__()
        self._data = None
        self._parameterized_data = None
        self._start = None
        self._end = None
        self._num_ctrlpts = None
        self._init_curve = None
        self._fit_curve = None

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def data(self):
        """
        Boundary data points

        :getter: Gets boundary point data
        :type: ndarray
        """
        return self._data

    @data.setter
    def data(self, data_array):
        # Sanitize inputs
        if not isinstance(data_array, np.ndarray):
            try:
                data_array = np.array(data_array)
            except Exception:
                print('Input data is not a numpy array and could not be cast')
                raise

        # Check dimension
        if data_array.ndim != 2:
            raise ValueError('Boundary data array must be 2D')

        # Check that data is either 2D or 3D
        if not (data_array.shape[-1] == 2 or data_array.shape[-1] == 3):
            raise ValueError('Boundary data must be in either R2 or R3')

        self._data = data_array

    @property
    def parameterized_data(self):
        """
        Parameterized data points.

        [X1, X2, X3, u value]

        :getter: Gets parameterized point data
        :type: ndarray
        """
        return self._parameterized_data

    def parameterize(self):
        pass

    @property
    def start(self):
        """
        Boundary data start point

        :getter: Gets curve start point
        :type: ndarray
        """
        return self._start

    @start.setter
    def start(self, point):
        # Sanitize inputs
        if not isinstance(point, np.ndarray):
            try:
                point = np.array(point)
            except Exception:
                print('Start point was not an array and could not be cast')
                raise

        # Check dimension
        if point.ndim != 1:
            raise ValueError('Start point array must be 1D')

        # Check point in 2D or 3D
        if not (len(point) == 2 or len(point) == 3):
            raise ValueError('Start point must be in R2 or R3')

        self._start = point

    @property
    def end(self):
        """
        Boundary data end point

        :getter: Gets curve end point
        :type: ndarray
        """
        return self._end

    @end.setter
    def end(self, point):
        # Sanitize inputs
        if not isinstance(point, np.ndarray):
            try:
                point = np.array(point)
            except Exception:
                print('End point was not an array and could not be cast')
                raise

        # Check dimension
        if point.ndim != 1:
            raise ValueError('End point array must be 1D')

        # Check point in 2D or 3D
        if not (len(point) == 2 or len(point) == 3):
            raise ValueError('End point must be in R2 or R3')

        self._end = point

    @property
    def num_ctrlpts(self):
        """
        Number of control points associated with boundary data curve

        :getter: Get number of control points
        :type: int
        """
        return self._num_ctrlpts

    @num_ctrlpts.setter
    def num_ctrlpts(self, num):
        # Sanitize input
        if not isinstance(num, int):
            try:
                num = int(num)
            except Exception:
                print('Number of control points was not an int and could not be cast')
                raise

        if num <= 3:
            raise ValueError('Number of control points must be 3 or more')

        self._num_ctrlpts = num
