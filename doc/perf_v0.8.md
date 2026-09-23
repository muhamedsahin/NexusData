# NexusData v0.8 performance notes

Machine: Windows (MinGW), Release, AVX2=1. Numbers from `bench_suite` (best of 3–50).

| Benchmark | Time | Throughput |
|-----------|------|------------|
| `byte_scan_scalar` (32 MiB) | 0.028 s | **1.11 GiB/s** |
| `byte_scan_dispatch` (32 MiB) | 0.0065 s | **4.82 GiB/s** (~4.3× scalar) |
| `csv_record_starts` (32 MiB, no quotes) | 0.076 s | 0.41 GiB/s |
| host memcpy (calibration probe 16 MiB) | — | **9.37 GiB/s** |
| `collate_stack` B=64 F=128 | 5 µs | **6.6 GiB/s** |
| DataLoader sync 4k×32 | 2.7 ms | 366 epochs/s |
| DataLoader workers=2 | 2.6 ms | 391 epochs/s |
| CSV load 20k×16 | ~2.7 s | I/O+parse bound (Windows) |

## Calibration (no CUDA on this run)

```
image_nbytes_threshold  = 16 MiB
tabular_nbytes_threshold = 32 MiB
assumed_pcie_gib_s      = 6.0
launch_overhead_us      = 40
```

Auto therefore stays on Host for typical image batches < 16 MiB when CUDA is absent.

## Optimizations shipped

1. **SWAR** scalar `find_bytes` (8-byte parallel compare).
2. **Quote-free** `find_csv_record_starts`: `memchr('"')` then SIMD newline scan.
3. **`collate_stack` / pad labels**: direct `memcpy` rows (no temporary `vector<NDArray>`).
4. **`calibrate_auto_device_policy`**: host memcpy probe + PCIe break-even thresholds.

## Claims vs theory

| Claim | Evidence |
|-------|----------|
| Dispatch beats scalar on AVX2 | Measured 4.3× on 32 MiB newline scan |
| Collate is memcpy-bound | ~6–7 GiB/s ≈ significant fraction of host memcpy |
| Workers help small InMemory | Mild gain (366→391) — dataset already RAM-resident |

Do not claim “fastest DataLoader” without cross-library benches on the same machine.
