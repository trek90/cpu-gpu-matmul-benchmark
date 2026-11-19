"""CPU/CUDA measurement checks and separately labelled synthetic policy tests."""

import math

import pytest
import torch

from measure import measure
from policy import Measurement, evaluate, fit


@pytest.mark.parametrize("mode", ["roundtrip", "resident"])
def test_real_cpu_measurement(mode):
    row = measure((11, 7, 13), mode, repeats=3, warmup=1, device="cpu")
    assert row.shape == (11, 7, 13)
    assert row.mode == mode
    assert math.isfinite(row.cpu_ms) and row.cpu_ms > 0
    assert row.gpu_ms is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
@pytest.mark.parametrize("mode", ["roundtrip", "resident"])
def test_real_cuda_measurement(mode):
    row = measure((37, 19, 31), mode, repeats=3, warmup=1, device="cuda")
    assert math.isfinite(row.cpu_ms) and row.cpu_ms > 0
    assert row.gpu_ms is not None and math.isfinite(row.gpu_ms) and row.gpu_ms > 0


@pytest.mark.parametrize("mode", ["roundtrip", "resident"])
def test_policy_without_gpu_evidence_uses_cpu(mode):
    rows = [Measurement(shape=(32, 32, 32), mode=mode, cpu_ms=0.03, gpu_ms=None)]
    policy = fit(rows)
    assert policy.choose((64, 64, 64), mode) == "cpu"


def synthetic_row(size, mode):
    """Fabricated monotonic timings; these are not observed hardware data."""
    launch_cost = {"roundtrip": 1.0, "resident": 0.1}[mode]
    work = (size / 128) ** 3
    return Measurement(
        shape=(size, size, size),
        mode=mode,
        cpu_ms=work,
        gpu_ms=launch_cost + work * 0.08,
    )


def test_synthetic_policy_separates_modes_on_unseen_shapes():
    modes = ("roundtrip", "resident")
    train = [synthetic_row(size, mode) for mode in modes for size in (32, 64, 96, 128, 160, 192, 256)]
    policy = fit(train)
    # 80 is an unseen shape where resident wins but roundtrip loses.
    assert policy.choose((80, 80, 80), "resident") == "cuda"
    assert policy.choose((80, 80, 80), "roundtrip") == "cpu"
    assert policy.choose((224, 224, 224), "roundtrip") == "cuda"
    assert policy.choose((40, 40, 40), "resident") == "cpu"


def test_synthetic_evaluation_accepts_heldout_measurements():
    modes = ("roundtrip", "resident")
    train = [synthetic_row(size, mode) for mode in modes for size in (32, 64, 96, 128, 160, 192, 256)]
    heldout = [synthetic_row(size, mode) for mode in modes for size in (40, 80, 224)]
    assert {row.shape for row in train}.isdisjoint(row.shape for row in heldout)
    assert evaluate(fit(train), heldout) == {
        "measured_rows": 6,
        "device_accuracy": 1.0,
        "mean_latency_ratio": 1.0,
        "max_latency_ratio": 1.0,
    }


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a host without CUDA")
def test_cuda_request_reports_missing_latency_on_cpu_only_host():
    row = measure((7, 5, 3), "roundtrip", repeats=2, warmup=0, device="cuda")
    assert row.cpu_ms > 0
    assert row.gpu_ms is None


def test_cpu_only_rows_are_not_used_as_crossover_ground_truth():
    rows = [Measurement(shape=(32, 64, 96), mode="resident", cpu_ms=0.01)]
    assert evaluate(fit(rows), rows) == {
        "measured_rows": 0,
        "device_accuracy": None,
        "mean_latency_ratio": None,
        "max_latency_ratio": None,
    }


def test_dispatch_executes_real_cpu_matmul():
    from dispatch import matmul
    from policy import fit

    a = torch.tensor([[2., 3.]])
    b = torch.tensor([[5.], [7.]])
    torch.testing.assert_close(matmul(a, b, fit([])), torch.tensor([[31.]]))
