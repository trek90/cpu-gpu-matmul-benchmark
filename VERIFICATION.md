# Verification

PyTorch 2.8.0+cpu, one CPU thread: 9 tests passed and 2 CUDA tests
were skipped because no CUDA device was available. The end-to-end
benchmark emitted 20 calibration rows and 12 held-out rows. In one run,
median CPU round-trip-path measurements ranged from about 7 microseconds
for 64 × 64 square inputs to 77 milliseconds for 1,536 × 1,536.
These are single-environment sample measurements, not portable results.

No GPU timing, hardware crossover, or routing accuracy was measured in
this run. Synthetic policy tests validate decision logic only.
