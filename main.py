import numpy as np

def residuals_squared(m, b, array):
    """
    Return the sum of the squares of the difference between the actual value and a lines predicted value
    """
    line = lambda x: (m*x) + b
    predicted_values = line(array[:, 0])
    residuals = array[:, 1] - predicted_values
    return np.dot(residuals, residuals)

def regress(array):
    """
    Find the M and B of the line that best fits an array of data structured as pairs of [x, y] points, eg: [[1, 1], [2, 2], [3, 3]]
    """
    assert array.ndim == 2

    zxi, zyi = array.sum(axis=0)
    n = array.shape[0]
    zxiyi = np.dot(array[:, 0], array[:, 1])
    zxizyi = zxi * zyi / n
    z_xisquared = np.dot(array[:, 0], array[:, 0])
    zxi_squared = zxi**2 / n
    m = (zxiyi - zxizyi) / ( z_xisquared - zxi_squared )
    b = (zyi - m * zxi) / n
