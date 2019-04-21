"""
.. module:: fileIO
    :platform: Unix, Windows
    :synopsis: Functions for handling reading/writing of point cloud and NURBS/B-spline data to and from files

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import spline


def write_curve_to_txt(curveinstance, filename):
    """
    Save Spline Curve parametrization data to text file.

    Saves degree, knot vector, and control points.

    :param curveinstance: Curve object to save
    :type curveinstance: spline.Curve() object
    :param filename: Path (may be relative or absolute) to destination file. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """
    with open(filename, 'r') as f:
        # Write degree information
        f.write('Degree - p:\n')
        f.write('{}\n'.format(curveinstance.degree))

        # Write knot vector information
        f.write('Number of Knots\n')
        f.write('{}\n'.format(len(curveinstance.knot_vector)))

        f.write('Knot Vector - U\n')
        for knot in range(curveinstance.knot_vector):
            f.write('{}\t'.format(curveinstance.knot_vector[knot]))
        f.write('\n')

        # Write control points info
        f.write('Number of Control Points\n')
        f.write('{}\n'.format(len(curveinstance.control_points)))

        f.write('Control Points\n')
        for ctrlpt in range(curveinstance.control_points):
            f.write('{}\t {}\t {}\n'.format(curveinstance.control_points[ctrlpt, 0],
                                            curveinstance.control_points[ctrlpt, 1],
                                            curveinstance.control_points[ctrlpt, 2]))


def read_curve_from_txt(curve, filename):
    """
    Read spline curve data from file and modify an instance of the spline.Curve() class

    :param curve: Curve object to define with data in filename
    :type curve: spline.Curve() object
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
        curve.degree = degree
        curve.control_points = control_points
        curve.knot_vector = knot_vector
