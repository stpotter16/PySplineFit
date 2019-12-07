#!/usr/bin/env python
"""
.. script:: synthetic_valve_example
    :platform: Unix, Windows
    :synopsis: Fit a spline surface to synthetic data and save results vtk
.. scriptauthor:: Sam Potter <spotter1642@gmail.com>
"""

import numpy as np
from pysplinefit import data

# Load data
top_data = np.loadtxt('ref_top.csv', delimiter=',')
bottom_data = np.loadtxt('ref_bot.csv', delimiter=',')
interior_data = np.loadtxt('ref_data.csv', delimiter=',')

# Fit the top boundary
# Setup top boundary instance
top = data.Boundary()
top.degree = 2
top.num_ctrlpts = 7

# Set the data and start/end
top.data = top_data
top.start = top_data[0, :]
top.end = top_data[-1, :]

# Set fit method
top.fit_method = 'fixed'

# Initialize curve
top.set_init_curve()

# Fit top curve
print('Fitting top boundary\n')
top.fit()
print('\n')

# Fit the bottom boundary
# Setup bottom boundary instance
bottom = data.Boundary()
bottom.degree = 2
bottom.num_ctrlpts = 7

# Set the data and start/end
bottom.data = bottom_data
bottom.start = bottom_data[0, :]
bottom.end = bottom_data[-1, :]

# Set fit method
bottom.fit_method = 'fixed'

# Initialize curve
bottom.set_init_curve()

# Fit bottom curve
print('Fitting bottom boundary\n')
bottom.fit()
print('\n')

# Setup the surface
surf = data.Interior()
surf.degree = 2
surf.num_ctrlpts = 3

# Set boundaries
surf.top_boundary = top
surf.bottom_boundary = bottom

# Set data
surf.data = interior_data

# Initialize surface
surf.set_init_surface()

# Fit surface
print('Fit interior data\n')
surf.fit()
print('\n')

surf.vtk('example.vtk')
