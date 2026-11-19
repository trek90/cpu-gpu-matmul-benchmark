# CPU vs. GPU Matrix Multiplication: Benchmark and Device Selection

When is it faster to multiply matrices on the CPU than to send them to a
GPU? Measure both paths, fit a small device selection policy, then score its
decisions on matrix shapes it never saw during calibration. Shapes are
listed as `(m, k, n)` for multiplying `[m, k] @ [k, n]`.

Two separate experiments:

- **Round trip:** two inputs start on the CPU and the result must end on
  the CPU. GPU latency includes both input copies and the output copy.
- **Resident:** inputs are already on the selected device and the result
  stays there. CPU and GPU timings are compute-path baselines, not the
  cost of migrating an already GPU-resident request to the CPU.

CPU measurements use the configured PyTorch intra-op thread count. GPU
timings wait for actual CUDA completion. The benchmark compares FP32 outputs
before accepting GPU timings. It reports median per-call wall-clock latency
after warm-up. These are machine-specific measurements, not portable
thresholds or throughput under concurrent load.

```sh
python -m pip install torch pytest
python run.py --threads 4 --json crossover.json
python -m pytest -q
```

Use a CUDA-enabled PyTorch build on an NVIDIA GPU to get crossover results.
Without one, the runner still measures real CPU calls but emits no GPU
timings, accuracy, or speedup. Synthetic unit-test fixtures exercise the
policy only; they are not performance data.

Read the code in this order: `run.py` defines the calibration and held-out
split, `measure.py` measures a shape, and `policy.py` decides which device
to prefer. A deployed router would recalibrate for its own hardware, dtype,
thread budget, and transfer path. Treat near ties with caution.

For actual round-trip dispatch, call `dispatch.matmul(a, b, fit(calibration))`
with two CPU tensors. It returns a CPU tensor. Use the resident policy to
compare device placements, not to route inputs that are already on CUDA.
