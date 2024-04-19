import math
import torch

def get_residual(x, q_x):
    """
    Compute the residual between the original tensor and the quantized tensor.
    """
    return x - q_x

def qsgd_b(x, b, is_correction=False):
    """
    Quantized Stochastic Gradient Descent with b-bit quantization.

    Args:
        x (torch.Tensor): The parameter to be quantized.
        b (int): The number of bits used for quantization.
        is_correction (bool): Whether to return the residual or not.

    Returns:
        torch.Tensor: The quantized parameter.
        torch.Tensor: The residual (if is_correction is True).
    """
    # Compute the quantization level
    # d = dimension of x
    d = x.numel()
    x_norm = torch.norm(x)
    w = 1 + min(math.sqrt(d) / (2 ** b), (d / (2 ** (2* b))))
    u = torch.rand_like(x)

    # Compute the quantized value
    sign_x = torch.sign(x)
    qsgd_b_x = (sign_x * (2 ** b * torch.abs(x) / x_norm + u)) / (2 * b * w + 1e-8)

    if is_correction:
        residual = get_residual(x, qsgd_b_x)
        return qsgd_b_x, residual

    return qsgd_b_x, None