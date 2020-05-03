"""
.. module:: base_geometry
    :platform: Unix, Windows
    :synopsis: Base classes for geometry
"""

from abc import ABC

class SplineGeometry(ABC):
    """ A base class for geometry objects

    Attributes
    ----------
    dimension : int
        The spatial dimension of the geometry object
    """

    def __init__(self, **kwargs):
        self._dimension = 0 if 'dimension' not in kwargs.keys() else kwargs['dimension']

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def dimension(self):
        return self._dimension


class SplineCurve(SplineGeometry):
    """ A base class for spline curves

    Attributes
    ----------
    degree : int
        The degree of the curve
    control_points : array
        The control points of the curve
    knot_vector : array
        The knot vector of the curve
    """

    def __init__(self, **kwargs):
        self._degree = None if 'degree' not in kwargs.keys() else kwargs['degree']
        self._control_points = None
        self._knot_vector = None
        super(SplineCurve, self).__init__(**kwargs)
    
    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        if degree < 1:
            raise ValueError('Degree must be greater than zero')
        self._degree = degree

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, ctrlpt_array):
        if self._degree is None:
            raise ValueError('Curve degree must be set before setting control points')

        if ctrlpt_array.shape[-1] != self.dimension:
            msg = 'Control points must be in R{}'.format(self.dimension)
            raise ValueError(msg)

    @property
    def knot_vector(self):
        return self._knot_vector

    @knot_vector.setter
    def knot_vector(self, kv):
        if self._degree is None:
            raise ValueError('Curve degree must be set befor setting knot vector')
        
        if self._control_points is None:
            raise ValueError('Curve control points must be set before knot vector')

        if self._check_knot_vector(kv):
            self._knot_vector = kv
    
    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def points(self, knots):
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def insert_knot(self, knot):
        """ Insert a single knot of specified value into the curves knot vector

        Parameters
        ----------
        knot: float
            Knot value to be inserted
        """
        pass

    @abc.abstractmethod
    def _check_knot_vector(self, kv):
        """ Check that knot vector is valid
        """
        pass


    class SplineSurface(SplineGeometry):
        """ A base class for spline surfaces

        Attributes
        ----------
        degree_u: int
            The degree of the surface in u direction
        degree_v : int
            The degree of the surface in the v direction
        control_points : array
            The control points of the surface
        knot_vector_u : array
            The knot vector of the surface in u direction
        knot_vector_v : array
            The knot voector of the surface in v direction
        """

    def __init__(self, **kwargs):
        self._degree_u = None if 'degree_u' not in kwargs.keys() else kwargs['degree_u']
        self._degree_v = None if 'degree_v' not in kwargs.keys() else kwargs['degree_v']
        self._num_control_points_u = None
        self._num_control_points_v = None
        self._control_points = None
        self._knot_vector_u = None
        self._knot_vector_v = None
        super(SplineCurve, self).__init__(**kwargs)

    @property
    def degree_u(self):
        return self._degree_u

    @degree_u.setter
    def degree_u(self, degree):
        if degree < 1:
            raise ValueError('Degree must be greater than zero')
        self._degree_u = degree

    @property
    def degree_v(self):
        return self._degree_v

    @degree_v.setter
    def degree_v(self, degree):
        if degree < 1:
            raise ValueError('Degree mvst be greater than zero')
        self._degree_v = degree

    @property
    def num_ctrlpts_u(self):
        return self._num_control_points_u

    @num_ctrlpts_u.setter
    def num_ctrlpts_u(self, num):
        if num < self.degree_u + 1:
            msg = ('Number of control points in u must be {} '
                   'for a surface with u degree of {}').format(self.degree_u + 1, self.degree_u)
            raise ValueError(msg)

        self._num_control_points_u = num

    @property
    def num_ctrlpts_v(self):
        return self._num_control_points_v

    @num_ctrlpts_v.setter
    def num_ctrlpts_v(self, num):
        if num < self.degree_v + 1:
            msg = ('Number of control points in v must be {} '
                   'for a surface with v degree of {}').format(self.degree_v + 1, self.degree_v)
            raise ValueError(msg)

        self._num_control_points_v = num

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, ctrlpt_array):
        if self._degree_u is None:
            raise ValueError('Surface degree u must be set before setting control points')

        if self._degree_v is None:
            raise ValueError('Surface degree v must be set before setting control points')

        if not ctrlpt_array.shape[-1] != self.dimension:
            raise ValueError('Control point points must be in either R{}'.format(self.dimension))

        self._control_points = ctrlpt_array

    @property
    def knot_vector_u(self):
        return self._knot_vector_u

    @knot_vector_u.setter
    def knot_vector_u(self, kv):
        if self._degree_u is None:
            raise ValueError('Surface degree in u direction must be set before setting knot vector')

        if self._control_points is None:
            raise ValueError("Surface control points must be set before setting knot vector")

        if self._num_ctrlpts_u is None:
            raise ValueError('Surface control point number in u must be set before setting knot vector')
        
        if self._check_knot_vector(kv, direction='u'):
            self._knot_vector_u = kv 

    @property
    def knot_vector_v(self):
        return self._knot_vector_v

    @knot_vector_v.setter
    def knot_vector_v(self, kv):
        if self._degree_v is None:
            raise ValueError('Surface degree in v direction must be set before setting knot vector')

        if self._control_points is None:
            raise ValueError("Surface control points must be set before setting knot vector")

        if self._num_ctrlpts_v is None:
            raise ValueError('Surface control point number in v must be set before setting knot vector')
        
        if self._check_knot_vector(kv, direction='v'):
            self._knot_vector_v = kv 

    @abc.abstractmethod
    def single_point(self, knot_u, knot_v):
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
        pass

    @abc.abstractmethod
    def points(self, knots):
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
        pass

    @abc.abstractmethod
    def derivatives(self, knot_u, knot_v, order_u, order_v, normalize=True):
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
        pass

    @abc.abstractmethod
    def _check_knot_vector(self, kv, direction='u'):
        """ Check that knot vector is valid
        """
        pass
