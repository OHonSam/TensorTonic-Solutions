import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    pmf = (1 - p)**(np.array(k) - 1) * p
    expected_mean = 1.0 / p
    return pmf, expected_mean