import numpy as np
from scipy.special import comb

def calc_binomial_pmf(n, p, k):
    n_factorial = np.prod(range(1, n + 1))
    k_factorial = np.prod(range(1, k + 1))
    # precision error: coeff = n_factorial * 1.0 / (k_factorial * np.prod(range(1, n - k + 1))) 
    coeff = comb(n, k)
    return coeff * p**k * (1-p)**(n-k)

def calc_binomial_cdf(n, p, k):
    cum_sum = 0.0
    for i in range(k + 1):
        cum_sum += calc_binomial_pmf(n, p, i)
    return cum_sum

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    return calc_binomial_pmf(n, p, k), calc_binomial_cdf(n, p, k)