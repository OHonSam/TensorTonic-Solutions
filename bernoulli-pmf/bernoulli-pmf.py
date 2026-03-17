import numpy as np

def calc_pmf(x, p):
    return p**x * (1 - p)**(1 - x)

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.array(x)
    mean = p
    var = p*(1-p)
    pmf = calc_pmf(x, p)

    return pmf, mean, var