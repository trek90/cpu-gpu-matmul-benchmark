"""Measure end-to-end PyTorch FP32 matrix multiplication latency."""

from __future__ import annotations

from collections.abc import Callable
from statistics import median
from time import perf_counter_ns

import torch
from torch import Tensor

from policy import Measurement


def _median_ms(
    operation: Callable[[], Tensor],
    synchronize: Callable[[], None],
    *,
    repeats: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        operation()
    synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter_ns()
        output = operation()
        synchronize()
        samples.append((perf_counter_ns() - started) / 1_000_000)
        del output
    return median(samples)


def _noop() -> None:
    pass


@torch.inference_mode()
def measure(
    shape: tuple[int, int, int],
    mode: str,
    *,
    repeats: int = 9,
    warmup: int = 3,
    device: str | torch.device = "cuda",
) -> Measurement:
    """Benchmark one (m×k) @ (k×n) FP32 shape using fixed inputs.

    CPU operands are already in host memory. For roundtrip timing, each
    iteration copies inputs from pageable host memory to CUDA, computes,
    and returns the output to CPU. For resident timing, copies happen
    before sampling and both operands and output remain on CUDA.
    All CUDA samples include host dispatch and completion synchronization.
    CUDA latency is None when CUDA is unavailable or device is CPU.
    PyTorch's existing CPU threading and matmul precision settings apply.
    """
    if len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("shape must contain three positive integers: (m, k, n)")
    if mode not in ("roundtrip", "resident"):
        raise ValueError("mode must be 'roundtrip' or 'resident'")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    m, k, n = shape
    generator = torch.Generator(device="cpu").manual_seed(426)
    a = torch.randn((m, k), generator=generator, dtype=torch.float32, device="cpu")
    b = torch.randn((k, n), generator=generator, dtype=torch.float32, device="cpu")

    def cpu() -> Tensor:
        return torch.mm(a, b)

    cpu_ms = _median_ms(cpu, _noop, repeats=repeats, warmup=warmup)
    if target.type == "cpu" or not torch.cuda.is_available():
        return Measurement(shape=tuple(shape), mode=mode, cpu_ms=cpu_ms, gpu_ms=None)
    if target.index is not None and target.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {target} does not exist")
    if mode == "resident":
        a_gpu, b_gpu = a.to(target), b.to(target)

        def gpu() -> Tensor:
            return torch.mm(a_gpu, b_gpu)
    else:
        def gpu() -> Tensor:
            return torch.mm(a.to(target), b.to(target)).cpu()

    # Correctness and an initial CUDA/library setup pass aren't timed.
    torch.testing.assert_close(gpu().cpu(), cpu(), rtol=1e-3, atol=1e-3)

    def sync() -> None:
        torch.cuda.synchronize(target)

    gpu_ms = _median_ms(gpu, sync, repeats=repeats, warmup=warmup)
    return Measurement(shape=tuple(shape), mode=mode, cpu_ms=cpu_ms, gpu_ms=gpu_ms)


@torch.inference_mode()
def execute(
    a: Tensor,
    b: Tensor,
    device: str | torch.device,
    *,
    return_cpu: bool = True,
) -> Tensor:
    """Run FP32 matrix multiplication on the selected device.

    Transfers only occur when needed. If returning a CUDA tensor, this
    function does not synchronize: normal PyTorch stream rules apply.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible two-dimensional matrices")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("both matrices must use float32")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output = torch.mm(a.to(target), b.to(target))
    return output.cpu() if return_cpu else output
