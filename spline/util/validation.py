"""
.. module:: base_geometry
    :platform: Unix, Windows
    :synopsis: Base classes for geometry
"""

def validate_knot(knot):
    """ Confirm a knot is in the range [0, 1]

    Parameters
    ----------
    knot : float
        Parameter to verify

    Returns
    -------
    bool
        Whether or not the knot is valid
    """

    return (0.0 <= knot <= 1.0)
