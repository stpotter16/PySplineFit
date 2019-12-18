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
    evaluated_points : array
        Coordinates of the points evaluated on the geometry
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
    def control_points(self, ctrlpy_array):
        if self._degree is None:
            raise ValueError('Curve degree must be set before setting control points')

        if ctrlpy_array.shape[-1] != self.dimension:
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
    def _check_knot_vector(self, kv):
        """ Check that knot vector is valid
        """
        pass
    
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
            Array of parametric points to evaluate
        
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
            Array of points and derivatives at specified not
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
        self._degree = None if 'degree' not in kwargs.keys() else kwargs['degree']
        self._control_points = None
        self._knot_vector = None
        super(SplineCurve, self).__init__(**kwargs)
 