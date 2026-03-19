import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    sample_size = len(x)
    sample_mean = np.average(x)
    standard_deviation = np.sqrt(np.sum((np.array(x) - sample_mean)**2) / (sample_size - 1))

    standard_error = standard_deviation / np.sqrt(sample_size)

    t_test = (sample_mean - mu0) / standard_error

    return t_test