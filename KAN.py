import numpy as np

"""
Personal attempt at KAN implementation

For experimental purposes in identifying how to fuse kans
"""



# Setup spline activation function
def cubic_spline(x, knots, coefficients):
    """
    Evaluate a cubic spline at given x values.
    :param x: Input value(s).
    :param knots: List of knot points.
    :param coefficients: Coefficients of the cubic polynomial segments.
    :return: Evaluated spline values.
    """
    # Ensure x is a numpy array
    x = np.array(x)
    result = np.zeros_like(x)

    # Find which interval each x belongs to
    for i in range(len(knots) - 1):
        mask = (x >= knots[i]) & (x < knots[i+1])
        if np.any(mask):
            x_segment = x[mask]
            a, b, c, d = coefficients[i]
            t = (x_segment - knots[i]) / (knots[i+1] - knots[i])
            result[mask] = (a * t**3 + b * t**2 + c * t + d)
    
    return result

#