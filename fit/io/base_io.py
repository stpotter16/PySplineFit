"""
.. module:: base_io
    :platform: Unix, Windows
    :synopsis: Base class for file reader/writers
"""

from abc import ABC
import os

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

    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write(self):
        raise NotImplementedError
