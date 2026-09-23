# NexusData architecture (v0.7)

## Hardening goals

1. **No abort on corrupt input** — parsers throw `nexusdata::Error` subclasses only.
2. **Actionable messages** — paths quoted, indices show `[0, N)`, CSV keeps `file:line:col`.
3. **Sanitizer-friendly** — optional ASan/UBSan/TSan via `NEXUSDATA_SANITIZER`.
4. **Fuzz surface** — in-memory NPY/WAV APIs + `fuzz::try_*` harness for CI smoke.

## Fuzz topology

```
libFuzzer (Clang)          portable CI
   fuzz_npy ──┐               │
   fuzz_wav ──┼── try_* ── parsers
              │               │
         fuzz_smoke (xorshift mutate, ctest)
```

## Design choices

| Choice | Rationale |
|--------|-----------|
| Portable `fuzz_smoke` always | Windows/MinGW often lack libFuzzer; still gate regressions |
| Catch only `Error` in harness | Unexpected `std::exception` fails the smoke test |
| Buffer APIs for NPY/WAV | Avoid tempfile I/O in tight fuzz loops |
| Header-only `fuzz/harness.hpp` | No extra link dep for harness helpers |

## Deferred

Continuous OSS-Fuzz integration, full corpus seeding for WebDataset/BMP, MSVC ASAN CI matrix, expanding clang-tidy to `-Werror` by default.
