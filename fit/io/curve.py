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

    def _json_reader(self):
        with open(self.file_handle, 'r') as f:
            data = json.load(f)
            return BSplineCurve.from_dict(data)


class CurveWriter(BaseWriter):
    """ A class for writing spline curves
    """
    def __init__(self, *args, **kwargs):
        self.supported_extensions = {
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

    def _json_writer(self):
        with open(self.file_handle, 'w') as f:
            data = self.spline.to_dict()
            json.dump(data, f)
