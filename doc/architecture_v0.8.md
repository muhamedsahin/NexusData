# NexusData architecture (v0.8)

## Goals

1. Measure hot paths with a lightweight harness (no Google Benchmark dependency).
2. Fix confirmed hotspots (byte scan, CSV record starts, collate).
3. Calibrate `AutoDevicePolicy` thresholds from host memcpy + PCIe break-even model.

## Calibration model

```
break_even_bytes ≈ 2 * assumed_pcie_gib_s * launch_overhead_s * 2^30
image_nbytes_threshold  ← break_even (or inflated if no CUDA)
tabular_nbytes_threshold ← ~2× image
```

`assumed_pcie_gib_s` defaults to **6.0** (effective Gen3-ish). Override on `AutoDevicePolicy` before `calibrate_auto_device_policy`.

## Deferred

Full GPU vs CPU microbench matrix on real PCIe, AVX-512/NEON scanners, persistent-worker pool reuse across epochs, OSS continuous bench dashboard.
