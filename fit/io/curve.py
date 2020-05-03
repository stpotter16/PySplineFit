"""
.. module:: curve
    :platform: Unix, Windows
    :synopsis: Reader/writer for curves
"""
import json
import os

import numpy as np

from fit.io.base_io import BaseReader, BaseWriter
from spline.model.bspline import BSplineCurve


class CurveReader(BaseReader):
    """ A class for reading spline curves
    """
    def __init__(self, *args, **kwargs):
        self.supported_extensions = {
            '.txt': self._text_reader,
            '.json': self._json_reader,
        }
        super(CurveReader, self).__init__(*args, **kwargs)

    def read(self):
        """ Read curve information from a file

        Returns
        -------
        spline : A SplineCurve instance
        """
        ext = os.path.splitext(self.file_handle)[1]
        reader = self.supported_extensions.get(ext, None)
        if reader is None:
            msg = (f'Reading files of extension type {ext} is not supported')
            raise ValueError(msg)
        reader()

    def _text_reader(self):
        with open(self.file_handle, 'r') as f:
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

    def _json_reader(self):
        with open(self.file_handle, 'r') as f:
            data = json.load(f)
            return BSplineCurve.from_dict(data)


class CurveWriter(BaseWriter):
    """ A class for writing spline curves
    """
    def __init__(self, *args, **kwargs):
        self.supported_extensions = {
            '.txt': self._text_writer,
            '.json': self._json_writer,
        }
        super(CurveWriter, self).__init__(*args, **kwargs)

    def write(self):
        """ Write spline curve information to a file
        """
        ext = os.path.splitext(self.file_handle)[1]
        writer = self.supported_extensions.get(ext, None)
        if writer is None:
            msg = (f'Writing files of extension type {ext} is not supported')
            raise ValueError(msg)
        writer()


    def _text_writer(self):
        """ Write to a text file """
        with open(self.file_handle, 'r') as f:
            # Write degree information
            f.write('Degree - p:\n')
            f.write('{}\n'.format(self.spline.degree))

            # Write knot vector information
            f.write('Number of Knots\n')
            f.write('{}\n'.format(len(self.spline.knot_vector)))

            f.write('Knot Vector - U\n')
            for knot in range(self.spline.knot_vector):
                f.write('{}\t'.format(self.spline.knot_vector[knot]))
            f.write('\n')

            # Write control points info
            f.write('Number of Control Points\n')
            f.write('{}\n'.format(len(self.spline.control_points)))

            f.write('Control Points\n')
            for ctrlpt in range(self.spline.control_points):
                f.write('{}\t {}\t {}\n'.format(self.spline.control_points[ctrlpt, 0],
                                                self.spline.control_points[ctrlpt, 1],
                                                self.spline.control_points[ctrlpt, 2]))

    def _json_writer(self):
        with open(self.file_handle, 'w') as f:
            data = self.spline.to_dict()
            json.dump(data, f)
