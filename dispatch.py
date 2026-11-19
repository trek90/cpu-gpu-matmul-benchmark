"""Run CPU-resident inputs through a calibrated round-trip policy."""

import torch

from measure import execute
from policy import Policy


def matmul(a: torch.Tensor, b: torch.Tensor, policy: Policy) -> torch.Tensor:
    """Choose a device for two CPU inputs and return a CPU result.

    Calibrate for the intended machine and CPU thread count. A CPU-only
    environment falls back to CPU even if the supplied policy picks CUDA.
    """
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("round-trip dispatch expects two CPU inputs")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible two-dimensional matrices")
    shape = (a.shape[0], a.shape[1], b.shape[1])
    device = policy.choose(shape, "roundtrip")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return execute(a, b, device)
