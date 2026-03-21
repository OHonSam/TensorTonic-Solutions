import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    input_shape = 2
    if len(x.shape) == 4:
        input_shape = 4
        N, C, H, W = x.shape
        x = np.transpose(x, (0,2,3,1))
        x = x.reshape(N * H * W, C)
    
    mu = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)

    x_norm = (x - mu) / np.sqrt(var + eps)
    y = x_norm * gamma + beta

    if input_shape == 4:
        y = y.reshape(N, H, W, C)
        y = np.transpose(y, (0,3,1,2))
        
    return y
    