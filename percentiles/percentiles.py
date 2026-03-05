import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.sort(x)
    q_np_array = np.array(q)
    q_index_down = q_np_array * (len(x) - 1) // 100
    q_index_up = (q_np_array * (len(x) - 1) + 99) // 100 
    q_index = (q_np_array * 1.0 * (len(x) - 1)) / 100

    res = x[q_index_down] + (q_index - q_index_down) * (x[q_index_up] - x[q_index_down])

    return res
    # for i in range(len(q)):
    #     idx_down = q_index_down[i] 
    #     idx_up = q_index_up[i]
    #     weight = q_index[i] - idx_down
    #     res[i] = x[idx_down] + (weight) * (x[idx_up] - x[idx_down])

    # return res
    