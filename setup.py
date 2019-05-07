#!/usr/bin/env python

"""
    Simple setup.py script for pysplinefit package
"""

from setuptools import setup, find_packages

# Meta-data
NAME = 'pysplinefit'
DESCRIPTION = 'Tools for fitting Spline curves and surfaces to unstructured data'
URL = 'https://github.com/stpotter16/PySplineFit'
EMAIL = 'spotter1642@gmail.com'
AUTHOR = 'Sam Potter'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = 1.0
REQUIRED = ['numpy', 'scipy', 'meshio']

# Call setup
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=['pysplinefit'],
    install_requires=REQUIRED,
    license='MIT',
    packages=find_packages()
)
