import numpy as np

def calc_poisson_pmf(lam, k):
    return np.exp(-lam) * lam**k / np.prod(range(1, k + 1))

def calc_poisson_cdf(lam, k):
    cum_sum = 0
    for k in range(0, k + 1):
        cum_sum += calc_poisson_pmf(lam, k)

    return cum_sum

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    return calc_poisson_pmf(lam, k), calc_poisson_cdf(lam, k)
    