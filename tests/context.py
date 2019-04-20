import os
import sys

# Add the library top directory to first position in $PATH variable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pysplinefit
import numpy as np
