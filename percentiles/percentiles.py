import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.sort(x)
    res = np.zeros_like(q, dtype=np.float64)
    q_index_down = np.array(q) * (len(x) - 1) // 100
    q_index_up = (np.array(q) * (len(x) - 1) + 99) // 100 
    q_index = (np.array(q) * 1.0 * (len(x) - 1)) / 100

    print(q_index_down)
    print(q_index)
    print(q_index_up)
    
    for i in range(len(q)):
        idx_down = q_index_down[i] 
        idx_up = q_index_up[i]
        weight = q_index[i] - idx_down
        res[i] = x[idx_down] + (weight) * (x[idx_up] - x[idx_down])

    return res
    