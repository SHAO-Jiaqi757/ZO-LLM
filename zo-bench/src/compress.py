import numpy as np

def get_residual(x, compressed_x):
    """
    Compute the residual between the original parameter and the compressed parameter.
    
    Args:
        x (numpy.ndarray): The original parameter.
        compressed_x (numpy.ndarray): The compressed parameter.
    
    Returns:
        numpy.ndarray: The residual between the original parameter and the compressed parameter.
    """
    return x - compressed_x

def qsgd_b(x, b, is_correction=False):
    """
    Quantized Stochastic Gradient Descent with b-bit quantization.
    
    Args:
        x (numpy.ndarray): The parameter to be quantized.
        b (int): The number of bits used for quantization.
    
    Returns:
        numpy.ndarray: The quantized parameter.
    """
    # Compute the quantization level
    # d = dimension of x
    d = x.size
    x_norm = np.linalg.norm(x)
    w = 1 + min(np.sqrt(d / (2 ** b)), d / (2 ** b))
    u = np.random.uniform(0, 1, size=x.shape)
    
    # Compute the quantized value
    sign_x = np.sign(x)
    qsgd_b_x = (sign_x * x_norm * (2 ** b * np.abs(x) / x_norm + u)) / (2 ** b * w + 1e-8)
    if is_correction:
        return qsgd_b_x, get_residual(x, qsgd_b_x)
    return qsgd_b_x, None


