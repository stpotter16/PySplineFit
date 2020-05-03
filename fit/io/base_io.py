"""
.. module:: base_io
    :platform: Unix, Windows
    :synopsis: Base class for file reader/writers
"""

from abc import ABC

class BaseFileIO(ABC):
    """ A base class for file reader/writers

    Attributes
    ----------
    file_handle : str
        Path to file for reading/writing
    """
    def __init__(self, file_handle, **kwargs):
        self.file_handle = file_handle
        super(BaseFileIO, self).__init__(**kwargs)


class BaseReader(BaseFileIO):
    """ A base class for file readers
    """
    @abc.abstractmethod
    def read(self):
        """ Read spline information from a file

        Returns
        -------
        spline : A SplineGeometry instance
        """
        raise NotImplementedError


class BaseWriter(BaseFileIO):
    """ A base class for file writers

    Attributes
    ----------
    spline : A SplineGeometry instance
    """
    def __init__(self, spline, *args, **kwargs):
        self.spline = spline
        super(BaseReader, self).__init__(*args, **kwargs)


    @abc.abstractmethod
    def write(self):
        """ Write spline information to a file
        """
        raise NotImplementedError
