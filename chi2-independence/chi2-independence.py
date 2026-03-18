import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    total_rows_count = np.sum(C, axis=0)
    total_cols_count = np.sum(C, axis=1)

    print(total_rows_count)
    print(total_cols_count.reshape(-1, 1))

    grand_total_count = np.sum(total_rows_count)

    print(grand_total_count)

    expected_frequencies = total_rows_count * total_cols_count.reshape(-1, 1) / grand_total_count

    print(expected_frequencies)
    
    chi2 = np.sum((C - expected_frequencies)**2 / expected_frequencies)

    print(chi2)

    return chi2, expected_frequencies