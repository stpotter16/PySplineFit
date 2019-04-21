"""
.. module:: fileIO
    :platform: Unix, Windows
    :synopsis: Functions for handling reading/writing of point cloud and NURBS/B-spline data to and from files

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def write_curve_to_txt(curveinstance, filename='curve.txt'):
    """
    Save Spline Curve parametrization data to text file.

    Saves degree, knot vector, and control points.

    :param curveinstance: Curve object to save
    :type curveinstance: spline.Curve() object
    :param filename: Optional. Path (may be relative or absolute) to destination file. Must include extension. Default
    'curve.txt'
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
