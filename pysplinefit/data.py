"""
.. module:: data
    :platform: Unix, Windows
    :synopsis: Contains high level classes for containing point cloud boundary and surface data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import spline
from . import np
from . import knots
from . import parameterize
from . import fitting
from . import fileIO


class Boundary:
    """
    Class container for boundary data
    """

    def __init__(self):
        self._degree = None
        self._data = None
        self._start = None
        self._end = None
        self._num_ctrlpts = None
        self._init_curve = None
        self._parameterized_data = None
        self._fit_curve = None

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

    @property
    def init_curve(self):
        """
        Initial curve between start and end points of boundary data


        :getter: Gets initial curve
        :type: spline.Curve()
        """
        return self._init_curve

    def set_init_curve(self):
        """
        Create and set an initial curve based on desired degree of boundary data curve fit
        :return: None
        """
        # Generate instance of curve class
        curve = spline.Curve()

        # Set degree
        curve.degree = self._degree

        # Set control points
        x_vals = np.linspace(self._start[0], self._end[0], self._degree + 1)
        y_vals = np.linspace(self._start[1], self._end[1], self._degree + 1)
        z_vals = np.linspace(self._start[2], self._end[2], self._degree + 1)

        init_ctrlpts = np.column_stack((x_vals, y_vals, z_vals))

        curve.control_points = init_ctrlpts

        # Generate knot vector
        init_knot_vector = knots.generate_uniform(self._degree, len(init_ctrlpts))

        curve.knot_vector = init_knot_vector

        self._init_curve = curve

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
        # Set the initial curve if not set, then parameterize
        if self._init_curve is None and self._fit_curve is None:
            self.set_init_curve()
            parameterized_data = parameterize.parameterize_curve(self._init_curve, self._data)
        # Parameterize to initial curve if not fit performed
        elif self._fit_curve is None:
            parameterized_data = parameterize.parameterize_curve(self._init_curve, self._data)
        # Parameterize to fit curve
        else:
            parameterized_data = parameterize.parameterize_curve(self._fit_curve, self._data)

        self._parameterized_data = parameterized_data

    @property
    def fit_curve(self):
        """
        Fit curve of specified degree and num_ctrlpts to data

        :getter: Gets fit_curve
        """
        return self._fit_curve

    def fit(self, logging=1):
        """
        Fit boundary data with curve.

        Set _final_curve and _parameterized_data properties

        :param logging: Option switch for log level. <=0 silent, =1 normal, >1 debug
        :type logging: int
        :return: None
        """
        # Check if init curve has been created. If not, create it
        if self._init_curve is None:
            self.set_init_curve()

        # Pass initial curve to fitting function
        final_fit = fitting.fit_curve_knot_insertion(self._init_curve, self._data, self._num_ctrlpts, logging=logging)

        num_ctrlpts = len(final_fit.knot_vector) - self.degree - 1

        # Set final curve property
        self._fit_curve = final_fit

        # Set number of control points
        self.num_ctrlpts = num_ctrlpts

        # Parameterize data
        self.parameterize()

    def save(self, name='boundary.txt'):
        """
        Save fit boundary curve object to file

        :param name: Optional. Path (relative or absolute) to file to save to. Default 'boundary.txt'
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

        # Check that fitting has been performed
        if self._fit_curve is None:
            raise ValueError('Fitting must be performed before save action')

        # Call fileIO function
        fileIO.write_curve_to_txt(self._fit_curve, name)

    def load(self, name):
        """
        Load fit boundary curve object from file and set degree, control points, and knot vector

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

        if not self._fit_curve is None:
            raise ValueError('A fit curve has already been defined')

        # Call fileIO function
        fileIO.read_curve_from_txt(self._fit_curve, name)


class Interior:
    """
    Class container for interior data
    """

    def __init__(self):
        self._degree = None
        self._data = None
        self._num_ctrlpts = None
        self._top_boundary = None
        self._bottom_boundary = None
        self._init_surface = None
        self._parameterized_data = None
        self._fit_surface = None

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def degree(self):
        """
        Degree of spline surface in v

        Degree of spline surface in u is accessed via the boundary

        :getter: Gets spline surface degree in v
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
    def data(self):
        """
        Interior data points

        :getter: Gets interior point data
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
            raise ValueError('Interior data array must be 2D')

        # Check that data is either 2D or 3D
        if not (data_array.shape[-1] == 2 or data_array.shape[-1] == 3):
            raise ValueError('Interior data must be in either R2 or R3')

        self._data = data_array

    @property
    def num_ctrlpts(self):
        """
        Number of control points associated with interior data surface

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

        if num < 3:
            raise ValueError('Number of control points must be 3 or more')

        self._num_ctrlpts = num

    @property
    def top_boundary(self):
        """
        Boundary curve corresponding to top of interior data

        :getter: Gets boundary curve
        :type: data.Boundary() object
        """
        return self._top_boundary

    @top_boundary.setter
    def top_boundary(self, boundary):
        # Sanitize input
        if not isinstance(boundary, Boundary):
            raise TypeError('Top Curve must be a Boundary object')

        if boundary.fit_curve is None:
            raise ValueError('Top Curve must have been fit before being set')

        self._top_boundary = boundary

    @property
    def bottom_boundary(self):
        """
        Boundary curve corresponding to bottom of interior data

        :getter: Gets boundary curve
        :type: data.Boundary() object
        """
        return self._bottom_boundary

    @bottom_boundary.setter
    def bottom_boundary(self, boundary):
        # Sanitize input
        if not isinstance(boundary, Boundary):
            raise TypeError('Bottom Curve must be a Boundary object')

        if boundary.fit_curve is None:
            raise ValueError('Bottom Curve must have been fit before being set')

        self._bottom_boundary = boundary

    @property
    def init_surface(self):
        """
        Initial surface between top and bottom boundary curves
        :getter: Gets initial surface
        :type: spline.Surface()
        """
        return self._init_surface

    def set_init_surface(self):
        """
        Creates and sets an initial surface between the top and bottom boundary curves
        :return: None
        """
        # Check inputs
        if self._top_boundary.degree != self._bottom_boundary.degree:
            raise ValueError('Boundary curves must have the same degree')

        if self._top_boundary.num_ctrlpts != self.bottom_boundary.num_ctrlpts:
            raise ValueError('Boundary curves must have the same number of control points')

        # Create surface
        initial_surf = spline.Surface()

        initial_surf.degree_u = self._bottom_boundary.degree
        initial_surf.degree_v = self._degree

        initial_surf.num_ctrlpts_u = self._bottom_boundary.num_ctrlpts
        initial_surf.num_ctrlpts_v = self._num_ctrlpts

        # Create control point
        surf_ctrlpt = np.zeros((initial_surf.num_ctrlpts_u * initial_surf.num_ctrlpts_v, 3))

        ctrlpt = 0
        for upt in range(0, initial_surf.num_ctrlpts_u):
            delta = self._top_boundary.fit_curve.control_points[upt, :] - \
                    self._bottom_boundary.fit_curve.control_points[upt, :]
            for vpt in range(0, self._num_ctrlpts):
                surf_ctrlpt[ctrlpt, :] = self._bottom_boundary.fit_curve.control_points[upt, :] +\
                                      vpt / (self._num_ctrlpts - 1) * delta
                ctrlpt += 1

        initial_surf.control_points = surf_ctrlpt

        # Set knot vectors
        initial_surf.knot_vector_u = self._bottom_boundary.fit_curve.knot_vector
        initial_surf.knot_vector_v = knots.generate_uniform(self._degree, self._num_ctrlpts)

        self._init_surface = initial_surf

    @property
    def parameterized_data(self):
        """
        Parameterized data points

        [X1, X2, X3, u, v]

        :getter: Gets parameterized point data
        :type: ndarray
        """
        return self._parameterized_data

    def parameterize(self):
        # Set initial surface if not set, then parameterize
        if self._init_surface is None and self._fit_surface is None:
            self.set_init_surface()
            parameterized_data = parameterize.parameterize_surface(self._init_surface, self._data)
        # Parameterize to initial curve if that's all the is set
        elif self._fit_surface is None:
            parameterized_data = parameterize.parameterize_surface(self._init_surface, self._data)
        # Parameterize to fit surface
        else:
            parameterized_data = parameterize.parameterize_surface(self._fit_surface, self._data)

        self._parameterized_data = parameterized_data

    @property
    def fit_surface(self):
        """
        Fit surface of specified degree and num_ctrlpts to data

        :getter: Gets fit_surface
        """
        return self._fit_surface

    def fit(self, logging=1):
        """
        Fit interior data with surface

        set _final_surface and _parameterized_data properties

        :param logging: Option switch for log level. <= silent, =1 normal, >1 debug
        :type logging: int
        :return: None
        """
        # Check if init surface has been created. If not, create it
        if self._init_surface is None:
            self.set_init_surface()

        # Pass initial surface to fitting function
        final_fit = fitting.fit_surface(self._init_surface, self._data, self._top_boundary, self._bottom_boundary,
                                        logging=logging)

        # Set final surface property
        self._fit_surface = final_fit

        # Parameterize data
        self.parameterize()

    def save(self, name='interior.txt'):
        """
        Save fit interior surface object to file

        :param name: Optional. Path (relative or absolute) to file to save to. Default 'interior.txt'
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

        # Check that fitting has been performed
        if self._fit_surface is None:
            raise ValueError('Fitting must be performed before save action')

        # Call fileIO function
        fileIO.write_surface_to_txt(self._fit_surface, name)

    def load(self, name):
        """
        Load fit interior surface object from file and set degree, control points, and knot vector

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

        if not self._fit_surface is None:
            raise ValueError('A fit curve has already been defined')

        # Call fileIO function
        fileIO.read_curve_from_txt(self._fit_surface, name)

    def vtk(self, name):
        """
        Write surface object to vtk file for visualization

        :param name: Path (relative or absolute) to file to write. Must include '.vtk' extension
        :type name: str
        :return:
        """

        # Check input
        if not isinstance(name, str):
            try:
                name = str(name)
            except Exception:
                print('Input file name was not a string and could not be cast')
                raise

        # Call fileIO function
        fileIO.surf_to_vtk(self._fit_surface, name)
