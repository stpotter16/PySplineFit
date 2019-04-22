"""
.. module:: fileIO
    :platform: Unix, Windows
    :synopsis: Functions for handling reading/writing of point cloud and NURBS/B-spline data to and from files

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import spatial
from . import meshio


def write_curve_to_txt(curve_instance, filename):
    """
    Save Spline Curve parametrization data to text file.

    Saves degree, knot vector, and control points.

    :param curve_instance: Curve object to save
    :type curve_instance: spline.Curve() object
    :param filename: Path (may be relative or absolute) to destination file. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """
    with open(filename, 'r') as f:
        # Write degree information
        f.write('Degree - p:\n')
        f.write('{}\n'.format(curve_instance.degree))

        # Write knot vector information
        f.write('Number of Knots\n')
        f.write('{}\n'.format(len(curve_instance.knot_vector)))

        f.write('Knot Vector - U\n')
        for knot in range(curve_instance.knot_vector):
            f.write('{}\t'.format(curve_instance.knot_vector[knot]))
        f.write('\n')

        # Write control points info
        f.write('Number of Control Points\n')
        f.write('{}\n'.format(len(curve_instance.control_points)))

        f.write('Control Points\n')
        for ctrlpt in range(curve_instance.control_points):
            f.write('{}\t {}\t {}\n'.format(curve_instance.control_points[ctrlpt, 0],
                                            curve_instance.control_points[ctrlpt, 1],
                                            curve_instance.control_points[ctrlpt, 2]))


def read_curve_from_txt(curve_instance, filename):
    """
    Read spline curve data from file and modify an instance of the spline.Curve() class

    :param curve_instance: Curve object to define with data in filename
    :type curve_instance: spline.Curve() object
    :param filename: Path (relative or absolute) to file containing curve data. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """

    with open(filename, 'r') as f:
        contents = [line.strip('\n') for line in f]

        # Pull degree
        degree = int(contents[1])

        # Get number of knots
        num_knots = int(contents[3])

        # Get knot values
        knot_vector = np.array(list(map(float, contents[5].split('\t'))))

        # Get number of control points
        num_ctrlpts = int(contents[7])

        # Get actual control points
        control_points = [list(map(float, contents[line].split())) for line in range(9, len(contents))]
        control_points = np.array(control_points)

        # Setup the curve
        curve_instance.degree = degree
        curve_instance.control_points = control_points
        curve_instance.knot_vector = knot_vector


def write_surface_to_txt(surface_instance, filename):
    """
    Save Spline Surface parametrization data to text file.

    Saves degree, knot vector, and control points.

    :param surface_instance: Surface object to save
    :type surface_instance: spline.Surface() object
    :param filename: Path (may be relative or absolute) to destination file. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """
    with open(filename, 'r') as f:
        # Write degree information
        f.write('Degree - p:\n')
        f.write('{}\n'.format(surface_instance.degree_u))

        f.write('Degree - q:\n')
        f.write('{}\n'.format(surface_instance.degree_v))

        # Write knot vector information
        f.write('Number of Knots in U\n')
        f.write('{}\n'.format(len(surface_instance.knot_vector_u)))

        f.write('Number of Knots in V\n')
        f.write('{}\n'.format(len(surface_instance.knot_vector_v)))

        f.write('Knot Vector - U\n')
        for knot in range(surface_instance.knot_vector_u):
            f.write('{}\t'.format(surface_instance.knot_vector_u[knot]))
        f.write('\n')

        f.write('Knot Vector - V\n')
        for knot in range(surface_instance.knot_vector_v):
            f.write('{}\t'.format(surface_instance.knot_vector_v[knot]))
        f.write('\n')

        # Write control points info
        f.write('Number of Control Points in U\n')
        f.write('{}\n'.format(surface_instance.num_ctrlpts_u))

        f.write('Number of Control Points in V\n')
        f.write('{}\n'.format(surface_instance.num_ctrlpts_v))

        f.write('Control Points\n')
        for ctrlpt in range(surface_instance.control_points):
            f.write('{}\t {}\t {}\n'.format(surface_instance.control_points[ctrlpt, 0],
                                            surface_instance.control_points[ctrlpt, 1],
                                            surface_instance.control_points[ctrlpt, 2]))


def read_surf_from_txt(surface_instance, filename):
    """
    Read spline surface data from file and modify an instance of the spline.Surface() class

    :param surface_instance: Surface object to define with data in filename
    :type surface_instance: spline.Surface() object
    :param filename: Path (relative or absolute) to file containing curve data. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """

    with open(filename, 'r') as f:
        contents = [line.strip('\n') for line in f]

        # Pull degree
        degree_u = int(contents[1])
        degree_v = int(contents[3])

        # Get number of knots
        num_knots_u = int(contents[5])
        num_knots_v = int(contents[7])

        # Get knot values
        knot_vector_u = np.array(list(map(float, contents[9].split('\t'))))
        knot_vector_v = np.array(list(map(float, contents[11].split('\t'))))

        # Get number of control points
        num_ctrlpts_u = int(contents[13])
        num_ctrlpts_v = int(contents[15])

        # Get actual control points
        control_points = [list(map(float, contents[line].split())) for line in range(17, len(contents))]
        control_points = np.array(control_points)

        # Setup the curve
        surface_instance.degree_u = degree_u
        surface_instance.degree_v = degree_v

        surface_instance.num_ctrlpts_u = num_ctrlpts_u
        surface_instance.num_ctrlpts_v = num_ctrlpts_v

        surface_instance.control_points = control_points

        surface_instance.knot_vector_u = knot_vector_u
        surface_instance.knot_vector_v = knot_vector_v


def surf_to_vtk(surface, filename, n_tri=100):
    """
    Wrap meshio library to write surface data to vtk

    :param surface: Spline surface to write to file
    :type surface: spline.Surface() object
    :param filename: Name of vtk file. Must include .vtk extension. Can be relative or aboslute
    :type filename: str
    :param n_tri: Optional. Number of triangles to form. Default 100
    :type n_tri: int
    :return: None
    """
    # Setup the start, end, and step size of parameterization
    start = 1 / n_tri
    end = 1
    size = int((end - start) * n_tri)
    spts = np.zeros((size * size, 5))

    # Loop through the paramete values in u and v
    pt = 0
    for uval in np.arrange(start, end, 1 / n_tri):
        for vval in np.arrange(start, end, 1/ n_tri):
            spts[pt, :] = np.append(surface.single_point(uval, vval), (uval, vval))

            pt += 1

    # Triangulate surface data on parameterization
    params = np.column_stack((spts[:, 3], spts[:, 4]))
    tri = spatial.Delaunay(params)
    simplicies = tri.simplicies

    # Assembly data for writing to VTK with meshio
    cells = {'triangle': simplicies}
    points = spts[:, 3]

    # Write to file
    mesh = meshio.Mesh(points, cells)
    meshio.write(filename, mesh)
