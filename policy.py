"""Device decisions from measured matmul crossover data."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median
from typing import Literal

Shape = tuple[int, int, int]
Device = Literal["cpu", "cuda"]
Mode = Literal["roundtrip", "resident"]
_MODES = ("roundtrip", "resident")


def _validate_shape(shape: Shape) -> None:
    if len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("shape must contain three positive integer dimensions (M, K, N)")


def _validate_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError("mode must be 'roundtrip' or 'resident'")


@dataclass(frozen=True, slots=True)
class Measurement:
    shape: Shape
    mode: str
    cpu_ms: float
    gpu_ms: float | None = None

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        _validate_mode(self.mode)
        if not math.isfinite(self.cpu_ms) or self.cpu_ms <= 0:
            raise ValueError("cpu_ms must be finite and positive")
        if self.gpu_ms is not None and (
            not math.isfinite(self.gpu_ms) or self.gpu_ms <= 0
        ):
            raise ValueError("gpu_ms must be missing or finite and positive")


@dataclass(frozen=True, slots=True)
class _Point:
    geometry: tuple[float, ...]
    log_speedup: float


def _geometry(shape: Shape) -> tuple[float, ...]:
    # Keep axes distinct: exchanging K for M changes arithmetic intensity.
    return tuple(math.log2(dim) for dim in shape)


@dataclass(frozen=True, slots=True)
class Policy:
    """A hardware- and workload-specific policy, not a universal threshold."""

    roundtrip: tuple[_Point, ...]
    resident: tuple[_Point, ...]

    def choose(self, shape: Shape, mode: str) -> Device:
        """Choose CUDA only when nearby, same-mode measurements support it."""
        _validate_shape(shape)
        _validate_mode(mode)
        points = self.roundtrip if mode == "roundtrip" else self.resident
        geometry = _geometry(shape)
        nearest = sorted(
            ((math.dist(geometry, p.geometry), i, p.log_speedup)
             for i, p in enumerate(points)),
        )[:3]
        if not nearest or nearest[0][0] > 2:
            return "cpu"
        exact = nearest[0][0] == 0
        neighbors = [row for row in nearest if row[0] <= 2]
        if exact:
            estimate = neighbors[0][2]
        else:
            # Distance is measured in log2 size; don't extrapolate from a
            # single isolated measurement or from disagreeing crossover data
            # on transfer-inclusive calls.
            if len(neighbors) < 2:
                return "cpu"
            weights = [1 / dist**2 for dist, _, _ in neighbors]
            estimate = (
                sum(weight * point[2] for weight, point in zip(weights, neighbors))
                / sum(weights)
            )
            if mode == "roundtrip" and any(log_ratio <= 0 for _, _, log_ratio in neighbors):
                return "cpu"
        margin = math.log(1.15 if mode == "roundtrip" else 1.05)
        return "cuda" if estimate >= margin else "cpu"


def fit(rows: Iterable[Measurement]) -> Policy:
    """Fit separate predictions in each mode using paired CPU/GPU timings."""
    grouped: dict[tuple[str, Shape], list[float]] = defaultdict(list)
    for row in rows:
        if row.gpu_ms is not None:
            grouped[row.mode, row.shape].append(
                math.log(row.cpu_ms) - math.log(row.gpu_ms)
            )
    points: dict[str, list[_Point]] = {mode: [] for mode in _MODES}
    for (mode, shape), values in sorted(grouped.items()):
        points[mode].append(_Point(_geometry(shape), median(values)))
    return Policy(tuple(points["roundtrip"]), tuple(points["resident"]))


def evaluate(policy: Policy, rows: Iterable[Measurement]) -> dict[str, int | float | None]:
    """Evaluate held-out paired timings; CPU-only rows never enter metrics."""
    count = correct = 0
    ratios: list[float] = []
    for row in rows:
        if row.gpu_ms is None:
            continue
        chosen = policy.choose(row.shape, row.mode)
        oracle = "cuda" if row.gpu_ms < row.cpu_ms else "cpu"
        count += 1
        correct += chosen == oracle
        faster = min(row.gpu_ms, row.cpu_ms)
        chosen_ms = row.gpu_ms if chosen == "cuda" else row.cpu_ms
        ratios.append(chosen_ms / faster)
    return {
        "measured_rows": count,
        "device_accuracy": correct / count if count else None,
        "mean_latency_ratio": sum(ratios) / count if count else None,
        "max_latency_ratio": max(ratios, default=None),
    }
