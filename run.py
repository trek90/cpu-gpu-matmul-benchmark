"""Measure CPU/CUDA matmul and evaluate a calibrated device picker."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from measure import measure
from policy import evaluate, fit

MODES = ("roundtrip", "resident")
CALIBRATION = (
    (32, 32, 32), (128, 128, 128), (384, 384, 384), (768, 768, 768),
    (1024, 1024, 1024), (2048, 2048, 2048), (16, 1024, 1024),
    (128, 512, 1024), (512, 1024, 64), (256, 256, 1536),
)
HELD_OUT = (
    (64, 64, 64), (256, 256, 256), (512, 512, 512), (1536, 1536, 1536),
    (32, 768, 1024), (256, 1024, 128),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Try CUDA by default; record CPU-only data if unavailable.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=Path, help="Save raw median timings and scores.")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("threads and repeats must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    cuda = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if args.device == "cuda" and not cuda:
        print("CUDA unavailable; GPU timings and device-picking scores will be omitted.")

    def collect(shapes):
        return [measure(shape, mode, repeats=args.repeats, warmup=args.warmup, device=device)
                for mode in MODES for shape in shapes]

    calibration = collect(CALIBRATION)
    picker = fit(calibration)
    held_out = collect(HELD_OUT)
    scores = {mode: evaluate(picker, [row for row in held_out if row.mode == mode])
              for mode in MODES}

    print("mode       m × k × n          CPU ms   CUDA ms  picker")
    for row in held_out:
        dims = "×".join(map(str, row.shape))
        gpu = "n/a" if row.gpu_ms is None else f"{row.gpu_ms:.4f}"
        print(f"{row.mode:10} {dims:17} {row.cpu_ms:7.4f} {gpu:>9}  "
              f"{picker.choose(row.shape, row.mode)}")
    if cuda:
        print(json.dumps(scores, indent=2))
    if args.json:
        report = {
            "platform": platform.platform(), "processor": platform.processor(),
            "torch_version": str(torch.__version__), "cpu_threads": args.threads,
            "repeats": args.repeats, "warmup": args.warmup,
            "cuda_device": torch.cuda.get_device_name() if cuda else None,
            "calibration": [asdict(row) for row in calibration],
            "held_out": [asdict(row) for row in held_out],
            "held_out_scores": scores if cuda else None,
        }
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
