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
            if normalize and not np.isclose(np.linalg.norm(val), 0.0) and row != 0:
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

        if num < 3:
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

        if num < 3:
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

    @property
    def knot_vector_u(self):
        """
        Knot vector of spline surface in u direction

        :getter: Gets spline surface knot vector in u direction
        :type: ndarray
        """
        return self._knot_vector_u

    @knot_vector_u.setter
    def knot_vector_u(self, knot_vector_array):

        # Check that degree has been set
        if self._degree_u is None:
            raise ValueError('Surface degree in u direction must be set before setting knot vector')

        # Check that control points are set
        if self._control_points is None:
            raise ValueError("Surface control points must be set before setting knot vector")

        # Check that num_ctrlpts_u is set
        if self._num_ctrlpts_u is None:
            raise ValueError('Surface control point number in u must be set before setting knot vector')

        # Check that input is okay
        if not isinstance(knot_vector_array, np.ndarray):
            try:
                knot_vector_array = np.array(knot_vector_array)  # Cast to array
            except Exception:
                print('Input value for knot vector was of invalid type and is unable to be cast to an array')
                raise

        # Check that knot vector is valid
        if not knots.check_knot_vector(self._degree_u, knot_vector_array, self._num_ctrlpts_u):
            raise ValueError('Knot vector is invalid')

        self._knot_vector_u = knot_vector_array

    @property
    def knot_vector_v(self):
        """
        Knot vector of spline surface in v direction

        :getter: Gets spline surface knot vector in v direction
        :type: ndarray
        """
        return self._knot_vector_v

    @knot_vector_v.setter
    def knot_vector_v(self, knot_vector_array):

        # Check that degree has been set
        if self._degree_v is None:
            raise ValueError('Surface degree in v direction must be set before setting knot vector')

        # Check that control points are set
        if self._control_points is None:
            raise ValueError("Surface control points must be set before setting knot vector")

        # Check that num_ctrlpts_v is set
        if self._num_ctrlpts_v is None:
            raise ValueError('Surface control point number in v must be set before setting knot vector')

        # Check that input is okay
        if not isinstance(knot_vector_array, np.ndarray):
            try:
                knot_vector_array = np.array(knot_vector_array)  # Cast to array
            except Exception:
                print('Input value for knot vector was of invalid type and is unable to be cast to an array')
                raise

        # Check that knot vector is valid
        if not knots.check_knot_vector(self._degree_v, knot_vector_array, self._num_ctrlpts_v):
            raise ValueError('Knot vector is invalid')

        self._knot_vector_v = knot_vector_array

    def single_point(self, u, v):
        """
        Evaluate a surface at a single parameteric point

        :param u: u parameter
        :type u: float
        :param v: v parameter
        :type v: float
        :return: value of surface at parameteric point as an array [X1, X2, X3]
        :rtype: ndarray
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
        u_span = knots.find_span(self._num_ctrlpts_u, self._degree_u, u, self._knot_vector_u)
        v_span = knots.find_span(self._num_ctrlpts_v, self._degree_v, v, self._knot_vector_v)

        # Evaluate basis functions
        basis_funs_u = basis.basis_functions(u_span, u, self._degree_u, self._knot_vector_u)
        basis_funs_v = basis.basis_functions(v_span, v, self._degree_v, self._knot_vector_v)

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
        """
        Evaluate the surface at multiple parameter coordinates

        :param knot_array: array of parameter values
        :type knot_array: ndarray
        :return: array of evaluated surface points
        :rtype: ndarray
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
        """
        Evaluate derivatives of the surface at specified parameteric location up to min(degree u, order u) and
        min(degree v, order v) in the u and v direction respectively

        :param u: u parameter
        :type u: float
        :param v: v parameter
        :type v: float
        :param order_u: max order of derivative in u
        :type order_u: int
        :param order_v: max order of deriviative in v
        :type order_v: int
        :param normalize: Optional. Boolean switch to control normalization of output derivatives
        :type normalize: bool
        :return: Tuple of point and derivatives at specified knot in each direction
        :rtype: tuple
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
        u_span = knots.find_span(self._num_ctrlpts_u, self._degree_u, u, self._knot_vector_u)
        v_span = knots.find_span(self._num_ctrlpts_v, self._degree_v, v, self._knot_vector_v)

        # Evaluate basis functions
        basis_funs_u_ders = basis.basis_function_ders(u_span, u, self._degree_u, self._knot_vector_u, max_order_u)
        basis_funs_v_ders = basis.basis_function_ders(v_span, v, self._degree_v, self._knot_vector_v, max_order_v)

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

    def save(self, name='surface.txt'):
        """
        Save surface object to file

        :param name: Optional. Path (relative or absolute) to file to save to. Default 'surface.txt'
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
        fileIO.write_surface_to_txt(self, name)

    def load(self, name):
        """
        Load surface object from file and set degree, control points, and knot vector

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
        fileIO.read_surf_from_txt(self, name)

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
        fileIO.surf_to_vtk(self, name)


