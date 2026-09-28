# lowbit benchmark results

Generated from `results/bench.json` by `lowbit report`. Do not edit by hand.

Measured 2026-09-28T23:09:29+00:00 on `Intel(R) Xeon(R) Processor @ 2.10GHz|4c` (ISA `avx512`, VNNI yes, LLC 260 MiB), numpy 2.4.6 with openblas 0.3.31.188.0 (SkylakeX). median of >=3 reps after 1 warm-up (t_min_s also recorded). Load average at start/end: 1.00/3.37. Contention control: wait for >= 90% idle CPU before each block; re-measure (up to 4x) when other processes used > 0.25 cores during a measurement.

Columns: median wall time per call; GFLOP/s = 2·M·N·K / time; weight GB/s = bytes of the weight format (packed codes + scales) / time; speedup is against `numpy fp32` (x @ W.T on pre-dequantized fp32 weights) at the same thread count and cache mode. `max rel err` is max|y - y_ref| / max|y_ref| against a float64 reference on the dequantized weights (W4A8 includes int8 activation quantization error).

## Headline

**Decode, M=1, N=11008, K=4096 (LLaMA-7B MLP up-projection), cold cache** — weights streamed from DRAM every call:

| method | weight bytes | 1T time | 1T weight GB/s | 1T vs numpy fp32 | 4T time | 4T weight GB/s | 4T vs numpy fp32 |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 172.0 MiB | 12.0 ms | 15.0 | 1.00× | 3339 µs | 54.0 | 1.00× |
| w4a16 avx2 | 22.8 MiB | 5904 µs | 4.1 | 2.04× | 1637 µs | 14.6 | 2.04× |
| w4a16 avx512 | 22.8 MiB | 3971 µs | 6.0 | 3.03× | 1070 µs | 22.4 | 3.12× |
| w4a8 avx2-maddubs | 22.8 MiB | 3029 µs | 7.9 | 3.97× | 810 µs | 29.6 | 4.13× |
| w4a8 avx512-vnni | 22.8 MiB | 2270 µs | 10.6 | 5.30× | 621 µs | 38.6 | 5.38× |
| w4a8 amx | 24.2 MiB | 2421 µs | 10.5 | 4.96× | 680 µs | 37.3 | 4.91× |
| mxfp4 avx2 | 22.8 MiB | 6274 µs | 3.8 | 1.92× | 1693 µs | 14.1 | 1.97× |
| mxfp4 avx512 | 22.8 MiB | 4399 µs | 5.4 | 2.73× | 866 µs | 27.6 | 3.85× |

**All shapes, 1 thread, hot cache** — GFLOP/s (speedup vs numpy fp32 on pre-dequantized weights):

| M | N | K | numpy fp32 | numpy deq+mm int4 | W4A16 scalar C | W4A16 (best SIMD) | W4A8 (AVX-512 VNNI) | W4A8 (AMX) | MXFP4 (best SIMD) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 13.4 | 0.5 (0.04×) | 2.8 (0.21×) | 41.5 (3.09×) | 85.4 (6.35×) | 62.3 (4.63×) | 41.3 (3.07×) |
| 1 | 11008 | 4096 | 12.2 | 0.6 (0.05×) | 2.9 (0.24×) | 45.8 (3.76×) | 99.6 (8.16×) | 63.2 (5.18×) | 45.7 (3.74×) |
| 1 | 4096 | 11008 | 11.3 | 0.6 (0.06×) | 3.0 (0.27×) | 43.6 (3.87×) | 93.8 (8.33×) | 65.9 (5.85×) | 44.6 (3.95×) |
| 16 | 4096 | 4096 | 38.5 | 9.3 (0.24×) | 3.1 (0.08×) | 101.6 (2.64×) | 209.9 (5.45×) | 590.7 (15.33×) | 94.9 (2.46×) |
| 64 | 4096 | 4096 | 90.2 | 34.8 (0.39×) | 3.1 (0.03×) | 85.1 (0.94×) | 251.5 (2.79×) | 635.7 (7.04×) | 87.5 (0.97×) |
| 16 | 11008 | 4096 | 41.6 | 9.1 (0.22×) | 3.1 (0.07×) | 104.5 (2.51×) | 234.8 (5.65×) | 668.9 (16.09×) | 90.8 (2.18×) |
| 64 | 11008 | 4096 | 90.0 | 29.0 (0.32×) | 3.1 (0.03×) | 83.3 (0.92×) | 214.8 (2.39×) | 639.5 (7.10×) | 93.7 (1.04×) |

**All shapes, 4 threads, hot cache** — GFLOP/s (speedup vs numpy fp32 on pre-dequantized weights):

| M | N | K | numpy fp32 | numpy deq+mm int4 | W4A16 scalar C | W4A16 (best SIMD) | W4A8 (AVX-512 VNNI) | W4A8 (AMX) | MXFP4 (best SIMD) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 46.8 | 0.5 (0.01×) | 11.3 (0.24×) | 141.7 (3.03×) | 246.4 (5.26×) | 198.2 (4.23×) | 129.7 (2.77×) |
| 1 | 11008 | 4096 | 53.5 | 0.7 (0.01×) | 11.5 (0.22×) | 170.5 (3.19×) | 266.3 (4.98×) | 211.5 (3.96×) | 147.6 (2.76×) |
| 1 | 4096 | 11008 | 49.2 | 0.7 (0.01×) | 11.8 (0.24×) | 172.0 (3.50×) | 276.7 (5.62×) | 217.1 (4.41×) | 151.8 (3.09×) |
| 16 | 4096 | 4096 | 141.4 | 13.7 (0.10×) | 11.8 (0.08×) | 371.7 (2.63×) | 668.4 (4.73×) | 1341.0 (9.48×) | 328.7 (2.32×) |
| 64 | 4096 | 4096 | 317.5 | 45.2 (0.14×) | 11.5 (0.04×) | 280.5 (0.88×) | 771.5 (2.43×) | 1523.2 (4.80×) | 320.9 (1.01×) |
| 16 | 11008 | 4096 | 155.3 | 10.7 (0.07×) | 12.9 (0.08×) | 409.5 (2.64×) | 768.9 (4.95×) | 1651.7 (10.64×) | 336.8 (2.17×) |
| 64 | 11008 | 4096 | 327.0 | 41.9 (0.13×) | 11.9 (0.04×) | 320.9 (0.98×) | 841.8 (2.57×) | 2164.8 (6.62×) | 321.7 (0.98×) |

Measured 2026-09-28T23:09:29+00:00 on `Intel(R) Xeon(R) Processor @ 2.10GHz|4c` (ISA `avx512`, VNNI yes, LLC 260 MiB), numpy 2.4.6 with openblas 0.3.31.188.0 (SkylakeX). median of >=3 reps after 1 warm-up (t_min_s also recorded). Load average at start/end: 1.00/3.37. Contention control: wait for >= 90% idle CPU before each block; re-measure (up to 4x) when other processes used > 0.25 cores during a measurement.

## llama.cpp comparison

**vs llama.cpp (ggml CPU backend), N=4096, K=14336, hot cache** — GFLOP/s (time per call):

| implementation | weights | M=1, 1T | M=8, 1T | M=1, 4T | M=8, 4T |
|---|---|---|---|---|---|
| llama.cpp/ggml | f32 | 7.1 (16.6 ms) | 40.7 (23.1 ms) | 39.3 (2987 µs) | 228.9 (4105 µs) |
| llama.cpp/ggml | q4_0 | 41.4 (2834 µs) | 75.5 (12.4 ms) | 149.4 (786 µs) | 279.9 (3357 µs) |
| llama.cpp/ggml | mxfp4 | 36.6 (3208 µs) | 32.6 (28.8 ms) | 141.4 (831 µs) | 151.8 (6189 µs) |
| lowbit | w4a16 | 46.5 (2528 µs) | 79.2 (11.9 ms) | 181.5 (647 µs) | 362.1 (2595 µs) |
| lowbit | w4a8 avx512-vnni | 96.7 (1214 µs) | 217.7 (4315 µs) | 331.7 (354 µs) | 696.0 (1350 µs) |
| lowbit | mxfp4 | 44.3 (2653 µs) | 87.7 (10.7 ms) | 160.6 (731 µs) | 330.1 (2846 µs) |
| lowbit | w4a8 amx | 65.9 (1782 µs) | 420.8 (2233 µs) | 226.4 (519 µs) | 1006.1 (934 µs) |

llama.cpp commit `1c4729414deba3bcacef6a5614e2ca04d069be3a`, measured 2026-09-28T23:10:47+00:00 with test-backend-ops perf -o MUL_MAT (CPU backend, default buffer type: no Q4_0 repacking / AMX extra buffers). ggml m/n = our N/M. ggml times are its own mean us/run; ours are medians. Both reuse the same weights every call (hot cache).

## All measurements

### M=1 N=4096 K=4096 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 2496 µs | 13.4 | 64.0 | 26.9 | 1.00× | 4.4e-07 |  |
| numpy deq+mm w4a16 | 69.4 ms | 0.5 | 8.5 | 0.1 | 0.04× | 4.4e-07 |  |
| numpy deq+mm mxfp4 | 104.6 ms | 0.3 | 8.5 | 0.1 | 0.02× | 4.4e-07 |  |
| w4a16 scalar | 12.0 ms | 2.8 | 8.5 | 0.7 | 0.21× | 2.0e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 1540 µs | 21.8 | 8.5 | 5.8 | 1.62× | 2.9e-07 | mr1 nr2 nb256 kb0 o0 |
| w4a16 avx512 | 808 µs | 41.5 | 8.5 | 11.0 | 3.09× | 2.3e-07 | mr1 nr1 nb64 kb0 o0 |
| w4a8 scalar | 8097 µs | 4.1 | 8.5 | 1.1 | 0.31× | 8.9e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 411 µs | 81.6 | 8.5 | 21.7 | 6.07× | 8.9e-03 | mr1 nr1 nb64 kb0 o0 |
| w4a8 avx512-vnni | 393 µs | 85.4 | 8.5 | 22.7 | 6.35× | 8.9e-03 | mr1 nr1 nb32 kb0 o0 |
| w4a8 amx | 539 µs | 62.3 | 9.0 | 17.5 | 4.63× | 8.9e-03 |  |
| mxfp4 scalar | 6727 µs | 5.0 | 8.5 | 1.3 | 0.37× | 4.4e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 1802 µs | 18.6 | 8.5 | 4.9 | 1.38× | 2.1e-07 | mr1 nr1 nb256 kb2048 o0 |
| mxfp4 avx512 | 813 µs | 41.3 | 8.5 | 11.0 | 3.07× | 2.1e-07 | mr1 nr4 nb32 kb0 o0 |

### M=1 N=4096 K=4096 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 717 µs | 46.8 | 64.0 | 93.6 | 1.00× | 4.4e-07 |  |
| numpy deq+mm w4a16 | 62.4 ms | 0.5 | 8.5 | 0.1 | 0.01× | 4.4e-07 |  |
| numpy deq+mm mxfp4 | 97.1 ms | 0.3 | 8.5 | 0.1 | 0.01× | 4.4e-07 |  |
| w4a16 scalar | 2964 µs | 11.3 | 8.5 | 3.0 | 0.24× | 2.0e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 422 µs | 79.5 | 8.5 | 21.1 | 1.70× | 2.9e-07 | mr1 nr2 nb64 kb0 o0 |
| w4a16 avx512 | 237 µs | 141.7 | 8.5 | 37.6 | 3.03× | 2.3e-07 | mr1 nr2 nb64 kb0 o0 |
| w4a8 scalar | 2217 µs | 15.1 | 8.5 | 4.0 | 0.32× | 8.9e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 146 µs | 229.7 | 8.5 | 61.0 | 4.91× | 8.9e-03 | mr1 nr2 nb64 kb0 o0 |
| w4a8 avx512-vnni | 136 µs | 246.4 | 8.5 | 65.4 | 5.26× | 8.9e-03 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 169 µs | 198.2 | 9.0 | 55.7 | 4.23× | 8.9e-03 |  |
| mxfp4 scalar | 1829 µs | 18.3 | 8.5 | 4.9 | 0.39× | 4.4e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 513 µs | 65.4 | 8.5 | 17.4 | 1.40× | 3.3e-07 | mr1 nr2 nb128 kb0 o0 |
| mxfp4 avx512 | 259 µs | 129.7 | 8.5 | 34.4 | 2.77× | 2.1e-07 | mr1 nr2 nb32 kb0 o0 |

### M=1 N=4096 K=4096 — cold cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 4502 µs | 7.5 | 64.0 | 14.9 | 1.00× | 4.4e-07 |  |
| w4a16 avx2 | 2282 µs | 14.7 | 8.5 | 3.9 | 1.97× | 2.9e-07 | mr1 nr2 nb256 kb0 o0 |
| w4a16 avx512 | 1147 µs | 29.3 | 8.5 | 7.8 | 3.92× | 2.3e-07 | mr1 nr1 nb64 kb0 o0 |
| w4a8 avx2-maddubs | 987 µs | 34.0 | 8.5 | 9.0 | 4.56× | 8.9e-03 | mr1 nr1 nb64 kb0 o0 |
| w4a8 avx512-vnni | 946 µs | 35.5 | 8.5 | 9.4 | 4.76× | 8.9e-03 | mr1 nr1 nb32 kb0 o0 |
| w4a8 amx | 1115 µs | 30.1 | 9.0 | 8.5 | 4.04× | 8.9e-03 |  |
| mxfp4 avx2 | 2522 µs | 13.3 | 8.5 | 3.5 | 1.79× | 2.1e-07 | mr1 nr1 nb256 kb2048 o0 |
| mxfp4 avx512 | 1518 µs | 22.1 | 8.5 | 5.9 | 2.97× | 2.1e-07 | mr1 nr4 nb32 kb0 o0 |

### M=1 N=4096 K=4096 — cold cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 1397 µs | 24.0 | 64.0 | 48.0 | 1.00× | 4.4e-07 |  |
| w4a16 avx2 | 669 µs | 50.2 | 8.5 | 13.3 | 2.09× | 2.9e-07 | mr1 nr2 nb64 kb0 o0 |
| w4a16 avx512 | 474 µs | 70.7 | 8.5 | 18.8 | 2.95× | 2.3e-07 | mr1 nr2 nb64 kb0 o0 |
| w4a8 avx2-maddubs | 355 µs | 94.4 | 8.5 | 25.1 | 3.93× | 8.9e-03 | mr1 nr2 nb64 kb0 o0 |
| w4a8 avx512-vnni | 265 µs | 126.8 | 8.5 | 33.7 | 5.28× | 8.9e-03 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 290 µs | 115.6 | 9.0 | 32.5 | 4.81× | 8.9e-03 |  |
| mxfp4 avx2 | 599 µs | 56.0 | 8.5 | 14.9 | 2.33× | 3.3e-07 | mr1 nr2 nb128 kb0 o0 |
| mxfp4 avx512 | 425 µs | 78.9 | 8.5 | 21.0 | 3.29× | 2.1e-07 | mr1 nr2 nb32 kb0 o0 |

### M=1 N=11008 K=4096 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 7388 µs | 12.2 | 172.0 | 24.4 | 1.00× | 4.7e-07 |  |
| numpy deq+mm w4a16 | 151.8 ms | 0.6 | 22.8 | 0.2 | 0.05× | 4.7e-07 |  |
| numpy deq+mm mxfp4 | 224.4 ms | 0.4 | 22.8 | 0.1 | 0.03× | 6.3e-07 |  |
| w4a16 scalar | 30.9 ms | 2.9 | 22.8 | 0.8 | 0.24× | 3.2e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 3746 µs | 24.1 | 22.8 | 6.4 | 1.97× | 2.9e-07 | mr1 nr2 nb128 kb0 o0 |
| w4a16 avx512 | 1967 µs | 45.8 | 22.8 | 12.2 | 3.76× | 2.4e-07 | mr1 nr2 nb64 kb0 o0 |
| w4a8 scalar | 21.1 ms | 4.3 | 22.8 | 1.1 | 0.35× | 8.5e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 1117 µs | 80.7 | 22.8 | 21.4 | 6.61× | 8.5e-03 | mr1 nr2 nb256 kb0 o0 |
| w4a8 avx512-vnni | 906 µs | 99.6 | 22.8 | 26.4 | 8.16× | 8.5e-03 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 1426 µs | 63.2 | 24.2 | 17.8 | 5.18× | 8.5e-03 |  |
| mxfp4 scalar | 17.8 ms | 5.1 | 22.8 | 1.3 | 0.42× | 4.7e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 4289 µs | 21.0 | 22.8 | 5.6 | 1.72× | 3.4e-07 | mr1 nr2 nb64 kb0 o0 |
| mxfp4 avx512 | 1973 µs | 45.7 | 22.8 | 12.1 | 3.74× | 2.3e-07 | mr1 nr2 nb64 kb0 o0 |

### M=1 N=11008 K=4096 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 1687 µs | 53.5 | 172.0 | 106.9 | 1.00× | 4.7e-07 |  |
| numpy deq+mm w4a16 | 132.8 ms | 0.7 | 22.8 | 0.2 | 0.01× | 4.7e-07 |  |
| numpy deq+mm mxfp4 | 227.0 ms | 0.4 | 22.8 | 0.1 | 0.01× | 6.3e-07 |  |
| w4a16 scalar | 7820 µs | 11.5 | 22.8 | 3.1 | 0.22× | 3.2e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 1034 µs | 87.2 | 22.8 | 23.2 | 1.63× | 2.9e-07 | mr1 nr2 nb16 kb0 o0 |
| w4a16 avx512 | 529 µs | 170.5 | 22.8 | 45.3 | 3.19× | 2.4e-07 | mr1 nr2 nb128 kb0 o0 |
| w4a8 scalar | 5618 µs | 16.1 | 22.8 | 4.3 | 0.30× | 8.5e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 324 µs | 278.2 | 22.8 | 73.9 | 5.21× | 8.5e-03 | mr1 nr2 nb128 kb0 o0 |
| w4a8 avx512-vnni | 339 µs | 266.3 | 22.8 | 70.7 | 4.98× | 8.5e-03 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 426 µs | 211.5 | 24.2 | 59.5 | 3.96× | 8.5e-03 |  |
| mxfp4 scalar | 4780 µs | 18.9 | 22.8 | 5.0 | 0.35× | 4.7e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 1164 µs | 77.4 | 22.8 | 20.6 | 1.45× | 3.4e-07 | mr1 nr2 nb128 kb0 o0 |
| mxfp4 avx512 | 611 µs | 147.6 | 22.8 | 39.2 | 2.76× | 2.3e-07 | mr1 nr1 nb64 kb0 o0 |

### M=1 N=11008 K=4096 — cold cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 12.0 ms | 7.5 | 172.0 | 15.0 | 1.00× | 4.7e-07 |  |
| w4a16 avx2 | 5904 µs | 15.3 | 22.8 | 4.1 | 2.04× | 2.9e-07 | mr1 nr2 nb128 kb0 o0 |
| w4a16 avx512 | 3971 µs | 22.7 | 22.8 | 6.0 | 3.03× | 2.4e-07 | mr1 nr2 nb64 kb0 o0 |
| w4a8 avx2-maddubs | 3029 µs | 29.8 | 22.8 | 7.9 | 3.97× | 8.5e-03 | mr1 nr2 nb256 kb0 o0 |
| w4a8 avx512-vnni | 2270 µs | 39.7 | 22.8 | 10.6 | 5.30× | 8.5e-03 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 2421 µs | 37.2 | 24.2 | 10.5 | 4.96× | 8.5e-03 |  |
| mxfp4 avx2 | 6274 µs | 14.4 | 22.8 | 3.8 | 1.92× | 3.4e-07 | mr1 nr2 nb64 kb0 o0 |
| mxfp4 avx512 | 4399 µs | 20.5 | 22.8 | 5.4 | 2.73× | 2.3e-07 | mr1 nr2 nb64 kb0 o0 |

### M=1 N=11008 K=4096 — cold cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 3339 µs | 27.0 | 172.0 | 54.0 | 1.00× | 4.7e-07 |  |
| w4a16 avx2 | 1637 µs | 55.1 | 22.8 | 14.6 | 2.04× | 2.9e-07 | mr1 nr2 nb16 kb0 o0 |
| w4a16 avx512 | 1070 µs | 84.2 | 22.8 | 22.4 | 3.12× | 2.4e-07 | mr1 nr2 nb128 kb0 o0 |
| w4a8 avx2-maddubs | 810 µs | 111.4 | 22.8 | 29.6 | 4.13× | 8.5e-03 | mr1 nr2 nb128 kb0 o0 |
| w4a8 avx512-vnni | 621 µs | 145.2 | 22.8 | 38.6 | 5.38× | 8.5e-03 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 680 µs | 132.5 | 24.2 | 37.3 | 4.91× | 8.5e-03 |  |
| mxfp4 avx2 | 1693 µs | 53.3 | 22.8 | 14.1 | 1.97× | 3.4e-07 | mr1 nr2 nb128 kb0 o0 |
| mxfp4 avx512 | 866 µs | 104.1 | 22.8 | 27.6 | 3.85× | 2.3e-07 | mr1 nr1 nb64 kb0 o0 |

### M=1 N=4096 K=11008 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 8000 µs | 11.3 | 172.0 | 22.5 | 1.00× | 5.2e-07 |  |
| numpy deq+mm w4a16 | 139.2 ms | 0.6 | 22.8 | 0.2 | 0.06× | 5.2e-07 |  |
| numpy deq+mm mxfp4 | 214.7 ms | 0.4 | 22.8 | 0.1 | 0.04× | 4.2e-07 |  |
| w4a16 scalar | 29.7 ms | 3.0 | 22.8 | 0.8 | 0.27× | 5.8e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 3797 µs | 23.8 | 22.8 | 6.3 | 2.11× | 5.5e-07 | mr1 nr2 nb32 kb0 o0 |
| w4a16 avx512 | 2067 µs | 43.6 | 22.8 | 11.6 | 3.87× | 4.2e-07 | mr1 nr2 nb16 kb0 o0 |
| w4a8 scalar | 21.8 ms | 4.1 | 22.8 | 1.1 | 0.37× | 1.0e-02 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 1129 µs | 79.9 | 22.8 | 21.2 | 7.09× | 1.0e-02 | mr1 nr1 nb16 kb0 o0 |
| w4a8 avx512-vnni | 961 µs | 93.8 | 22.8 | 24.9 | 8.33× | 1.0e-02 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 1369 µs | 65.9 | 24.2 | 18.5 | 5.85× | 1.0e-02 |  |
| mxfp4 scalar | 17.9 ms | 5.0 | 22.8 | 1.3 | 0.45× | 1.0e-06 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 4568 µs | 19.7 | 22.8 | 5.2 | 1.75× | 6.0e-07 | mr1 nr4 nb32 kb0 o0 |
| mxfp4 avx512 | 2023 µs | 44.6 | 22.8 | 11.8 | 3.95× | 4.1e-07 | mr1 nr2 nb128 kb0 o0 |

### M=1 N=4096 K=11008 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 1833 µs | 49.2 | 172.0 | 98.4 | 1.00× | 5.2e-07 |  |
| numpy deq+mm w4a16 | 123.5 ms | 0.7 | 22.8 | 0.2 | 0.01× | 5.2e-07 |  |
| numpy deq+mm mxfp4 | 202.9 ms | 0.4 | 22.8 | 0.1 | 0.01× | 4.2e-07 |  |
| w4a16 scalar | 7621 µs | 11.8 | 22.8 | 3.1 | 0.24× | 5.8e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 1040 µs | 86.7 | 22.8 | 23.0 | 1.76× | 5.5e-07 | mr1 nr2 nb32 kb0 o0 |
| w4a16 avx512 | 524 µs | 172.0 | 22.8 | 45.7 | 3.50× | 4.2e-07 | mr1 nr4 nb32 kb0 o0 |
| w4a8 scalar | 5554 µs | 16.2 | 22.8 | 4.3 | 0.33× | 1.0e-02 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 331 µs | 272.5 | 22.8 | 72.4 | 5.54× | 1.0e-02 | mr1 nr4 nb32 kb0 o0 |
| w4a8 avx512-vnni | 326 µs | 276.7 | 22.8 | 73.5 | 5.62× | 1.0e-02 | mr1 nr4 nb128 kb0 o0 |
| w4a8 amx | 415 µs | 217.1 | 24.2 | 61.1 | 4.41× | 1.0e-02 |  |
| mxfp4 scalar | 4709 µs | 19.2 | 22.8 | 5.1 | 0.39× | 1.0e-06 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 1185 µs | 76.1 | 22.8 | 20.2 | 1.55× | 6.0e-07 | mr1 nr4 nb16 kb0 o0 |
| mxfp4 avx512 | 594 µs | 151.8 | 22.8 | 40.3 | 3.09× | 4.1e-07 | mr1 nr4 nb64 kb0 o0 |

### M=1 N=4096 K=11008 — cold cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 12.1 ms | 7.5 | 172.0 | 14.9 | 1.00× | 5.2e-07 |  |
| w4a16 avx2 | 4956 µs | 18.2 | 22.8 | 4.8 | 2.44× | 5.5e-07 | mr1 nr2 nb32 kb0 o0 |
| w4a16 avx512 | 3105 µs | 29.0 | 22.8 | 7.7 | 3.89× | 4.2e-07 | mr1 nr2 nb16 kb0 o0 |
| w4a8 avx2-maddubs | 2606 µs | 34.6 | 22.8 | 9.2 | 4.63× | 1.0e-02 | mr1 nr1 nb16 kb0 o0 |
| w4a8 avx512-vnni | 2362 µs | 38.2 | 22.8 | 10.1 | 5.11× | 1.0e-02 | mr1 nr1 nb128 kb0 o0 |
| w4a8 amx | 2432 µs | 37.1 | 24.2 | 10.4 | 4.96× | 1.0e-02 |  |
| mxfp4 avx2 | 5183 µs | 17.4 | 22.8 | 4.6 | 2.33× | 6.0e-07 | mr1 nr4 nb32 kb0 o0 |
| mxfp4 avx512 | 3144 µs | 28.7 | 22.8 | 7.6 | 3.84× | 4.1e-07 | mr1 nr2 nb128 kb0 o0 |

### M=1 N=4096 K=11008 — cold cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 3052 µs | 29.5 | 172.0 | 59.1 | 1.00× | 5.2e-07 |  |
| w4a16 avx2 | 1350 µs | 66.8 | 22.8 | 17.7 | 2.26× | 5.5e-07 | mr1 nr2 nb32 kb0 o0 |
| w4a16 avx512 | 806 µs | 111.9 | 22.8 | 29.7 | 3.79× | 4.2e-07 | mr1 nr4 nb32 kb0 o0 |
| w4a8 avx2-maddubs | 601 µs | 150.1 | 22.8 | 39.9 | 5.08× | 1.0e-02 | mr1 nr4 nb32 kb0 o0 |
| w4a8 avx512-vnni | 568 µs | 158.6 | 22.8 | 42.1 | 5.37× | 1.0e-02 | mr1 nr4 nb128 kb0 o0 |
| w4a8 amx | 778 µs | 115.9 | 24.2 | 32.6 | 3.92× | 1.0e-02 |  |
| mxfp4 avx2 | 1447 µs | 62.3 | 22.8 | 16.6 | 2.11× | 6.0e-07 | mr1 nr4 nb16 kb0 o0 |
| mxfp4 avx512 | 839 µs | 107.5 | 22.8 | 28.6 | 3.64× | 4.1e-07 | mr1 nr4 nb64 kb0 o0 |

### M=16 N=4096 K=4096 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 13.9 ms | 38.5 | 64.0 | 4.8 | 1.00× | 3.4e-07 |  |
| numpy deq+mm w4a16 | 57.9 ms | 9.3 | 8.5 | 0.2 | 0.24× | 3.4e-07 |  |
| numpy deq+mm mxfp4 | 97.0 ms | 5.5 | 8.5 | 0.1 | 0.14× | 4.2e-07 |  |
| w4a16 scalar | 172.9 ms | 3.1 | 8.5 | 0.1 | 0.08× | 3.0e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 10.7 ms | 50.4 | 8.5 | 0.8 | 1.31× | 2.8e-07 | mr4 nr1 nb128 kb0 o0 |
| w4a16 avx512 | 5282 µs | 101.6 | 8.5 | 1.7 | 2.64× | 1.5e-07 | mr4 nr2 nb128 kb2048 o0 |
| w4a8 scalar | 122.8 ms | 4.4 | 8.5 | 0.1 | 0.11× | 8.5e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 3422 µs | 156.9 | 8.5 | 2.6 | 4.07× | 8.5e-03 | mr8 nr1 nb256 kb0 o0 |
| w4a8 avx512-vnni | 2558 µs | 209.9 | 8.5 | 3.5 | 5.45× | 8.5e-03 | mr4 nr4 nb128 kb0 o0 |
| w4a8 amx | 909 µs | 590.7 | 9.0 | 10.4 | 15.33× | 8.5e-03 |  |
| mxfp4 scalar | 101.4 ms | 5.3 | 8.5 | 0.1 | 0.14× | 3.5e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 10.7 ms | 50.3 | 8.5 | 0.8 | 1.31× | 2.1e-07 | mr4 nr1 nb256 kb2048 o0 |
| mxfp4 avx512 | 5657 µs | 94.9 | 8.5 | 1.6 | 2.46× | 1.6e-07 | mr4 nr4 nb128 kb2048 o0 |

### M=16 N=4096 K=4096 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 3796 µs | 141.4 | 64.0 | 17.7 | 1.00× | 3.4e-07 |  |
| numpy deq+mm w4a16 | 39.3 ms | 13.7 | 8.5 | 0.2 | 0.10× | 3.4e-07 |  |
| numpy deq+mm mxfp4 | 74.6 ms | 7.2 | 8.5 | 0.1 | 0.05× | 4.2e-07 |  |
| w4a16 scalar | 45.7 ms | 11.8 | 8.5 | 0.2 | 0.08× | 3.0e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 2840 µs | 189.0 | 8.5 | 3.1 | 1.34× | 1.9e-07 | mr4 nr1 nb256 kb2048 o0 |
| w4a16 avx512 | 1444 µs | 371.7 | 8.5 | 6.2 | 2.63× | 1.5e-07 | mr4 nr2 nb256 kb2048 o0 |
| w4a8 scalar | 33.4 ms | 16.1 | 8.5 | 0.3 | 0.11× | 8.5e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 1134 µs | 473.6 | 8.5 | 7.9 | 3.35× | 8.5e-03 | mr4 nr2 nb256 kb0 o0 |
| w4a8 avx512-vnni | 803 µs | 668.4 | 8.5 | 11.1 | 4.73× | 8.5e-03 | mr8 nr2 nb128 kb0 o0 |
| w4a8 amx | 400 µs | 1341.0 | 9.0 | 23.6 | 9.48× | 8.5e-03 |  |
| mxfp4 scalar | 25.0 ms | 21.5 | 8.5 | 0.4 | 0.15× | 3.5e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 2901 µs | 185.1 | 8.5 | 3.1 | 1.31× | 2.1e-07 | mr4 nr1 nb128 kb2048 o0 |
| mxfp4 avx512 | 1634 µs | 328.7 | 8.5 | 5.5 | 2.32× | 1.6e-07 | mr4 nr2 nb32 kb2048 o0 |

### M=64 N=4096 K=4096 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 23.8 ms | 90.2 | 64.0 | 2.8 | 1.00× | 3.9e-07 |  |
| numpy deq+mm w4a16 | 61.6 ms | 34.8 | 8.5 | 0.1 | 0.39× | 3.9e-07 |  |
| numpy deq+mm mxfp4 | 95.4 ms | 22.5 | 8.5 | 0.1 | 0.25× | 4.6e-07 |  |
| w4a16 scalar | 704.0 ms | 3.1 | 8.5 | 0.0 | 0.03× | 2.8e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 53.8 ms | 39.9 | 8.5 | 0.2 | 0.44× | 3.3e-07 | mr2 nr1 nb64 kb0 o0 |
| w4a16 avx512 | 25.2 ms | 85.1 | 8.5 | 0.4 | 0.94× | 2.2e-07 | mr2 nr4 nb32 kb0 o1 |
| w4a8 scalar | 485.0 ms | 4.4 | 8.5 | 0.0 | 0.05× | 8.7e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 13.5 ms | 159.3 | 8.5 | 0.7 | 1.77× | 8.7e-03 | mr8 nr1 nb128 kb0 o0 |
| w4a8 avx512-vnni | 8538 µs | 251.5 | 8.5 | 1.0 | 2.79× | 8.7e-03 | mr4 nr4 nb128 kb0 o1 |
| w4a8 amx | 3378 µs | 635.7 | 9.0 | 2.8 | 7.04× | 8.7e-03 |  |
| mxfp4 scalar | 386.3 ms | 5.6 | 8.5 | 0.0 | 0.06× | 4.0e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 52.6 ms | 40.8 | 8.5 | 0.2 | 0.45× | 2.2e-07 | mr4 nr4 nb64 kb2048 o0 |
| mxfp4 avx512 | 24.5 ms | 87.5 | 8.5 | 0.4 | 0.97× | 1.7e-07 | mr4 nr4 nb256 kb2048 o0 |

### M=64 N=4096 K=4096 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 6763 µs | 317.5 | 64.0 | 9.9 | 1.00× | 3.9e-07 |  |
| numpy deq+mm w4a16 | 47.5 ms | 45.2 | 8.5 | 0.2 | 0.14× | 3.9e-07 |  |
| numpy deq+mm mxfp4 | 85.3 ms | 25.2 | 8.5 | 0.1 | 0.08× | 4.6e-07 |  |
| w4a16 scalar | 187.0 ms | 11.5 | 8.5 | 0.0 | 0.04× | 2.8e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 12.1 ms | 178.0 | 8.5 | 0.7 | 0.56× | 1.6e-07 | mr4 nr1 nb64 kb1024 o0 |
| w4a16 avx512 | 7655 µs | 280.5 | 8.5 | 1.2 | 0.88× | 1.4e-07 | mr2 nr4 nb256 kb1024 o0 |
| w4a8 scalar | 135.3 ms | 15.9 | 8.5 | 0.1 | 0.05× | 8.7e-03 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 4152 µs | 517.2 | 8.5 | 2.1 | 1.63× | 8.7e-03 | mr8 nr1 nb32 kb0 o0 |
| w4a8 avx512-vnni | 2784 µs | 771.5 | 8.5 | 3.2 | 2.43× | 8.7e-03 | mr4 nr4 nb64 kb0 o0 |
| w4a8 amx | 1410 µs | 1523.2 | 9.0 | 6.7 | 4.80× | 8.7e-03 |  |
| mxfp4 scalar | 103.4 ms | 20.8 | 8.5 | 0.1 | 0.07× | 4.0e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 13.9 ms | 154.6 | 8.5 | 0.6 | 0.49× | 3.2e-07 | mr4 nr4 nb32 kb0 o0 |
| mxfp4 avx512 | 6693 µs | 320.9 | 8.5 | 1.3 | 1.01× | 2.4e-07 | mr4 nr4 nb16 kb0 o0 |

### M=16 N=11008 K=4096 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 34.7 ms | 41.6 | 172.0 | 5.2 | 1.00× | 4.1e-07 |  |
| numpy deq+mm w4a16 | 158.6 ms | 9.1 | 22.8 | 0.2 | 0.22× | 4.1e-07 |  |
| numpy deq+mm mxfp4 | 234.0 ms | 6.2 | 22.8 | 0.1 | 0.15× | 5.4e-07 |  |
| w4a16 scalar | 465.1 ms | 3.1 | 22.8 | 0.1 | 0.07× | 2.9e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 29.2 ms | 49.3 | 22.8 | 0.8 | 1.19× | 3.9e-07 | mr4 nr1 nb256 kb0 o0 |
| w4a16 avx512 | 13.8 ms | 104.5 | 22.8 | 1.7 | 2.51× | 1.7e-07 | mr4 nr2 nb32 kb2048 o0 |
| w4a8 scalar | 325.1 ms | 4.4 | 22.8 | 0.1 | 0.11× | 1.1e-02 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 8832 µs | 163.4 | 22.8 | 2.7 | 3.93× | 1.1e-02 | mr8 nr1 nb64 kb0 o0 |
| w4a8 avx512-vnni | 6144 µs | 234.8 | 22.8 | 3.9 | 5.65× | 1.1e-02 | mr4 nr4 nb64 kb0 o0 |
| w4a8 amx | 2157 µs | 668.9 | 24.2 | 11.8 | 16.09× | 1.1e-02 |  |
| mxfp4 scalar | 300.8 ms | 4.8 | 22.8 | 0.1 | 0.12× | 4.7e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 28.4 ms | 50.7 | 22.8 | 0.8 | 1.22× | 2.4e-07 | mr4 nr1 nb32 kb2048 o0 |
| mxfp4 avx512 | 15.9 ms | 90.8 | 22.8 | 1.5 | 2.18× | 2.0e-07 | mr4 nr4 nb128 kb2048 o0 |

### M=16 N=11008 K=4096 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 9290 µs | 155.3 | 172.0 | 19.4 | 1.00× | 4.1e-07 |  |
| numpy deq+mm w4a16 | 135.0 ms | 10.7 | 22.8 | 0.2 | 0.07× | 4.1e-07 |  |
| numpy deq+mm mxfp4 | 215.1 ms | 6.7 | 22.8 | 0.1 | 0.04× | 5.4e-07 |  |
| w4a16 scalar | 112.2 ms | 12.9 | 22.8 | 0.2 | 0.08× | 2.9e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 7288 µs | 198.0 | 22.8 | 3.3 | 1.27× | 2.2e-07 | mr4 nr1 nb128 kb2048 o0 |
| w4a16 avx512 | 3524 µs | 409.5 | 22.8 | 6.8 | 2.64× | 1.7e-07 | mr4 nr2 nb64 kb2048 o0 |
| w4a8 scalar | 86.1 ms | 16.8 | 22.8 | 0.3 | 0.11× | 1.1e-02 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 2536 µs | 569.0 | 22.8 | 9.4 | 3.66× | 1.1e-02 | mr8 nr2 nb128 kb0 o0 |
| w4a8 avx512-vnni | 1877 µs | 768.9 | 22.8 | 12.8 | 4.95× | 1.1e-02 | mr4 nr4 nb64 kb0 o1 |
| w4a8 amx | 874 µs | 1651.7 | 24.2 | 29.0 | 10.64× | 1.1e-02 |  |
| mxfp4 scalar | 71.5 ms | 20.2 | 22.8 | 0.3 | 0.13× | 4.7e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 7574 µs | 190.5 | 22.8 | 3.2 | 1.23× | 2.4e-07 | mr4 nr1 nb32 kb2048 o0 |
| mxfp4 avx512 | 4284 µs | 336.8 | 22.8 | 5.6 | 2.17× | 2.3e-07 | mr4 nr4 nb32 kb0 o1 |

### M=64 N=11008 K=4096 — hot cache, 1 thread

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 64.1 ms | 90.0 | 172.0 | 2.8 | 1.00× | 4.3e-07 |  |
| numpy deq+mm w4a16 | 198.8 ms | 29.0 | 22.8 | 0.1 | 0.32× | 4.3e-07 |  |
| numpy deq+mm mxfp4 | 256.4 ms | 22.5 | 22.8 | 0.1 | 0.25× | 5.0e-07 |  |
| w4a16 scalar | 1840.7 ms | 3.1 | 22.8 | 0.0 | 0.03× | 2.9e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 124.6 ms | 46.3 | 22.8 | 0.2 | 0.51× | 2.6e-07 | mr4 nr1 nb16 kb2048 o0 |
| w4a16 avx512 | 69.3 ms | 83.3 | 22.8 | 0.3 | 0.92× | 2.4e-07 | mr2 nr4 nb32 kb0 o0 |
| w4a8 scalar | 1317.8 ms | 4.4 | 22.8 | 0.0 | 0.05× | 1.0e-02 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 35.7 ms | 161.5 | 22.8 | 0.7 | 1.79× | 1.0e-02 | mr8 nr1 nb16 kb0 o0 |
| w4a8 avx512-vnni | 26.9 ms | 214.8 | 22.8 | 0.9 | 2.39× | 1.0e-02 | mr4 nr4 nb256 kb0 o1 |
| w4a8 amx | 9025 µs | 639.5 | 24.2 | 2.8 | 7.10× | 1.0e-02 |  |
| mxfp4 scalar | 1213.9 ms | 4.8 | 22.8 | 0.0 | 0.05× | 4.7e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 142.5 ms | 40.5 | 22.8 | 0.2 | 0.45× | 1.8e-07 | mr4 nr4 nb256 kb1024 o0 |
| mxfp4 avx512 | 61.6 ms | 93.7 | 22.8 | 0.4 | 1.04× | 1.8e-07 | mr4 nr4 nb128 kb2048 o0 |

### M=64 N=11008 K=4096 — hot cache, 4 threads

| method | time | GFLOP/s | weight MiB | weight GB/s | vs numpy fp32 | max rel err | tile params |
|---|---|---|---|---|---|---|---|
| numpy fp32 | 17.6 ms | 327.0 | 172.0 | 10.2 | 1.00× | 4.3e-07 |  |
| numpy deq+mm w4a16 | 137.9 ms | 41.9 | 22.8 | 0.2 | 0.13× | 4.3e-07 |  |
| numpy deq+mm mxfp4 | 220.9 ms | 26.1 | 22.8 | 0.1 | 0.08× | 5.0e-07 |  |
| w4a16 scalar | 484.7 ms | 11.9 | 22.8 | 0.0 | 0.04× | 2.9e-07 | mr0 nr0 nb0 kb0 o0 |
| w4a16 avx2 | 39.3 ms | 146.9 | 22.8 | 0.6 | 0.45× | 3.5e-07 | mr2 nr1 nb64 kb0 o0 |
| w4a16 avx512 | 18.0 ms | 320.9 | 22.8 | 1.3 | 0.98× | 2.4e-07 | mr2 nr4 nb16 kb0 o0 |
| w4a8 scalar | 350.9 ms | 16.4 | 22.8 | 0.1 | 0.05× | 1.0e-02 | mr0 nr0 nb0 kb0 o0 |
| w4a8 avx2-maddubs | 11.8 ms | 489.1 | 22.8 | 2.0 | 1.50× | 1.0e-02 | mr2 nr1 nb256 kb0 o0 |
| w4a8 avx512-vnni | 6856 µs | 841.8 | 22.8 | 3.5 | 2.57× | 1.0e-02 | mr4 nr4 nb32 kb0 o0 |
| w4a8 amx | 2666 µs | 2164.8 | 24.2 | 9.5 | 6.62× | 1.0e-02 |  |
| mxfp4 scalar | 298.2 ms | 19.4 | 22.8 | 0.1 | 0.06× | 4.7e-07 | mr0 nr0 nb0 kb0 o0 |
| mxfp4 avx2 | 42.0 ms | 137.3 | 22.8 | 0.6 | 0.42× | 3.6e-07 | mr4 nr4 nb16 kb0 o0 |
| mxfp4 avx512 | 17.9 ms | 321.7 | 22.8 | 1.3 | 0.98× | 2.6e-07 | mr4 nr4 nb64 kb0 o0 |

## Autotuner

Best tile parameters found per (kernel, ISA, threads, shape), and the gain over the library default (mr=4 or 1, nr=2, nb=64, whole K, order 0).

| key | params | configs tried | tuned GFLOP/s | gain vs default |
|---|---|---|---|---|
| `mxfp4|avx2|t1|16x11008x4096` | mr4 nr1 nb32 kb2048 o0 | 41 | 55.0 | 1.50× |
| `mxfp4|avx2|t1|16x4096x4096` | mr4 nr1 nb256 kb2048 o0 | 41 | 55.4 | 1.34× |
| `mxfp4|avx2|t1|1x11008x4096` | mr1 nr2 nb64 kb0 o0 | 18 | 23.4 | 1.11× |
| `mxfp4|avx2|t1|1x4096x11008` | mr1 nr4 nb32 kb0 o0 | 18 | 21.8 | 1.07× |
| `mxfp4|avx2|t1|1x4096x4096` | mr1 nr1 nb256 kb2048 o0 | 18 | 19.5 | 0.94× |
| `mxfp4|avx2|t1|64x11008x4096` | mr4 nr4 nb256 kb1024 o0 | 41 | 41.7 | 1.19× |
| `mxfp4|avx2|t1|64x4096x4096` | mr4 nr4 nb64 kb2048 o0 | 41 | 42.5 | 1.20× |
| `mxfp4|avx2|t4|16x11008x4096` | mr4 nr1 nb32 kb2048 o0 | 41 | 214.4 | 1.40× |
| `mxfp4|avx2|t4|16x4096x4096` | mr4 nr1 nb128 kb2048 o0 | 41 | 215.1 | 1.62× |
| `mxfp4|avx2|t4|1x11008x4096` | mr1 nr2 nb128 kb0 o0 | 18 | 72.7 | 0.98× |
| `mxfp4|avx2|t4|1x4096x11008` | mr1 nr4 nb16 kb0 o0 | 18 | 77.1 | 0.98× |
| `mxfp4|avx2|t4|1x4096x4096` | mr1 nr2 nb128 kb0 o0 | 18 | 67.4 | 1.04× |
| `mxfp4|avx2|t4|64x11008x4096` | mr4 nr4 nb16 kb0 o0 | 41 | 157.0 | 1.42× |
| `mxfp4|avx2|t4|64x4096x4096` | mr4 nr4 nb32 kb0 o0 | 41 | 161.6 | 1.82× |
| `mxfp4|avx512|t1|16x11008x4096` | mr4 nr4 nb128 kb2048 o0 | 99 | 103.7 | 1.44× |
| `mxfp4|avx512|t1|16x4096x4096` | mr4 nr4 nb128 kb2048 o0 | 99 | 106.2 | 1.35× |
| `mxfp4|avx512|t1|1x11008x4096` | mr1 nr2 nb64 kb0 o0 | 46 | 49.4 | 1.09× |
| `mxfp4|avx512|t1|1x4096x11008` | mr1 nr2 nb128 kb0 o0 | 46 | 47.8 | 1.05× |
| `mxfp4|avx512|t1|1x4096x14336` | mr1 nr4 nb64 kb0 o0 | 46 | 48.9 | 1.53× |
| `mxfp4|avx512|t1|1x4096x4096` | mr1 nr4 nb32 kb0 o0 | 46 | 46.6 | 1.11× |
| `mxfp4|avx512|t1|64x11008x4096` | mr4 nr4 nb128 kb2048 o0 | 99 | 94.4 | 1.84× |
| `mxfp4|avx512|t1|64x4096x4096` | mr4 nr4 nb256 kb2048 o0 | 99 | 96.1 | 1.91× |
| `mxfp4|avx512|t1|8x4096x14336` | mr4 nr4 nb256 kb0 o1 | 99 | 95.2 | 2.01× |
| `mxfp4|avx512|t4|16x11008x4096` | mr4 nr4 nb32 kb0 o1 | 99 | 385.9 | 1.37× |
| `mxfp4|avx512|t4|16x4096x4096` | mr4 nr2 nb32 kb2048 o0 | 99 | 368.8 | 1.17× |
| `mxfp4|avx512|t4|1x11008x4096` | mr1 nr1 nb64 kb0 o0 | 46 | 170.0 | 1.02× |
| `mxfp4|avx512|t4|1x4096x11008` | mr1 nr4 nb64 kb0 o0 | 46 | 163.7 | 0.96× |
| `mxfp4|avx512|t4|1x4096x14336` | mr1 nr2 nb128 kb0 o0 | 46 | 187.9 | 1.15× |
| `mxfp4|avx512|t4|1x4096x4096` | mr1 nr2 nb32 kb0 o0 | 46 | 137.8 | 1.07× |
| `mxfp4|avx512|t4|64x11008x4096` | mr4 nr4 nb64 kb0 o0 | 99 | 352.2 | 1.93× |
| `mxfp4|avx512|t4|64x4096x4096` | mr4 nr4 nb16 kb0 o0 | 99 | 364.6 | 2.83× |
| `mxfp4|avx512|t4|8x4096x14336` | mr4 nr4 nb16 kb0 o1 | 99 | 408.1 | 1.29× |
| `w4a16|avx2|t1|16x11008x4096` | mr4 nr1 nb256 kb0 o0 | 41 | 56.0 | 1.38× |
| `w4a16|avx2|t1|16x4096x4096` | mr4 nr1 nb128 kb0 o0 | 41 | 57.1 | 1.37× |
| `w4a16|avx2|t1|1x11008x4096` | mr1 nr2 nb128 kb0 o0 | 18 | 26.6 | 1.00× |
| `w4a16|avx2|t1|1x4096x11008` | mr1 nr2 nb32 kb0 o0 | 18 | 25.6 | 1.01× |
| `w4a16|avx2|t1|1x4096x4096` | mr1 nr2 nb256 kb0 o0 | 18 | 25.6 | 1.18× |
| `w4a16|avx2|t1|64x11008x4096` | mr4 nr1 nb16 kb2048 o0 | 41 | 51.5 | 1.44× |
| `w4a16|avx2|t1|64x4096x4096` | mr2 nr1 nb64 kb0 o0 | 41 | 42.3 | 1.19× |
| `w4a16|avx2|t4|16x11008x4096` | mr4 nr1 nb128 kb2048 o0 | 41 | 217.1 | 1.34× |
| `w4a16|avx2|t4|16x4096x4096` | mr4 nr1 nb256 kb2048 o0 | 41 | 191.3 | 1.26× |
| `w4a16|avx2|t4|1x11008x4096` | mr1 nr2 nb16 kb0 o0 | 18 | 89.2 | 1.11× |
| `w4a16|avx2|t4|1x4096x11008` | mr1 nr2 nb32 kb0 o0 | 18 | 68.2 | 1.03× |
| `w4a16|avx2|t4|1x4096x4096` | mr1 nr2 nb64 kb0 o0 | 18 | 94.8 | 1.05× |
| `w4a16|avx2|t4|64x11008x4096` | mr2 nr1 nb64 kb0 o0 | 41 | 162.2 | 1.17× |
| `w4a16|avx2|t4|64x4096x4096` | mr4 nr1 nb64 kb1024 o0 | 41 | 193.5 | 1.49× |
| `w4a16|avx512|t1|16x11008x4096` | mr4 nr2 nb32 kb2048 o0 | 99 | 111.8 | 1.30× |
| `w4a16|avx512|t1|16x4096x4096` | mr4 nr2 nb128 kb2048 o0 | 99 | 109.5 | 1.25× |
| `w4a16|avx512|t1|1x11008x4096` | mr1 nr2 nb64 kb0 o0 | 46 | 52.8 | 1.08× |
| `w4a16|avx512|t1|1x4096x11008` | mr1 nr2 nb16 kb0 o0 | 46 | 52.4 | 1.11× |
| `w4a16|avx512|t1|1x4096x14336` | mr1 nr4 nb256 kb0 o0 | 46 | 55.3 | 1.77× |
| `w4a16|avx512|t1|1x4096x4096` | mr1 nr1 nb64 kb0 o0 | 46 | 47.4 | 0.93× |
| `w4a16|avx512|t1|64x11008x4096` | mr2 nr4 nb32 kb0 o0 | 99 | 84.9 | 1.48× |
| `w4a16|avx512|t1|64x4096x4096` | mr2 nr4 nb32 kb0 o1 | 99 | 90.1 | 1.54× |
| `w4a16|avx512|t1|8x4096x14336` | mr2 nr4 nb32 kb0 o0 | 99 | 89.6 | 1.67× |
| `w4a16|avx512|t4|16x11008x4096` | mr4 nr2 nb64 kb2048 o0 | 99 | 428.9 | 3.56× |
| `w4a16|avx512|t4|16x4096x4096` | mr4 nr2 nb256 kb2048 o0 | 99 | 381.1 | 1.22× |
| `w4a16|avx512|t4|1x11008x4096` | mr1 nr2 nb128 kb0 o0 | 46 | 173.2 | 1.01× |
| `w4a16|avx512|t4|1x4096x11008` | mr1 nr4 nb32 kb0 o0 | 46 | 193.9 | 1.10× |
| `w4a16|avx512|t4|1x4096x14336` | mr1 nr4 nb32 kb0 o0 | 46 | 195.2 | 0.97× |
| `w4a16|avx512|t4|1x4096x4096` | mr1 nr2 nb64 kb0 o0 | 46 | 152.4 | 1.07× |
| `w4a16|avx512|t4|64x11008x4096` | mr2 nr4 nb16 kb0 o0 | 99 | 350.0 | 1.65× |
| `w4a16|avx512|t4|64x4096x4096` | mr2 nr4 nb256 kb1024 o0 | 99 | 322.2 | 2.40× |
| `w4a16|avx512|t4|8x4096x14336` | mr4 nr2 nb32 kb2048 o0 | 99 | 413.7 | 1.36× |
| `w4a8|avx2|t1|16x11008x4096` | mr8 nr1 nb64 kb0 o0 | 41 | 181.6 | 1.20× |
| `w4a8|avx2|t1|16x4096x4096` | mr8 nr1 nb256 kb0 o0 | 41 | 169.7 | 1.12× |
| `w4a8|avx2|t1|1x11008x4096` | mr1 nr2 nb256 kb0 o0 | 18 | 84.7 | 1.00× |
| `w4a8|avx2|t1|1x4096x11008` | mr1 nr1 nb16 kb0 o0 | 18 | 97.2 | 1.14× |
| `w4a8|avx2|t1|1x4096x4096` | mr1 nr1 nb64 kb0 o0 | 18 | 90.9 | 1.25× |
| `w4a8|avx2|t1|64x11008x4096` | mr8 nr1 nb16 kb0 o0 | 41 | 166.9 | 1.17× |
| `w4a8|avx2|t1|64x4096x4096` | mr8 nr1 nb128 kb0 o0 | 41 | 178.5 | 1.27× |
| `w4a8|avx2|t4|16x11008x4096` | mr8 nr2 nb128 kb0 o0 | 41 | 626.2 | 1.20× |
| `w4a8|avx2|t4|16x4096x4096` | mr4 nr2 nb256 kb0 o0 | 41 | 524.8 | 1.11× |
| `w4a8|avx2|t4|1x11008x4096` | mr1 nr2 nb128 kb0 o0 | 18 | 290.7 | 1.07× |
| `w4a8|avx2|t4|1x4096x11008` | mr1 nr4 nb32 kb0 o0 | 18 | 292.0 | 1.01× |
| `w4a8|avx2|t4|1x4096x4096` | mr1 nr2 nb64 kb0 o0 | 18 | 240.0 | 1.05× |
| `w4a8|avx2|t4|64x11008x4096` | mr2 nr1 nb256 kb0 o0 | 41 | 484.3 | 1.34× |
| `w4a8|avx2|t4|64x4096x4096` | mr8 nr1 nb32 kb0 o0 | 41 | 542.3 | 1.13× |
| `w4a8|avx512|t1|16x11008x4096` | mr4 nr4 nb64 kb0 o0 | 99 | 250.5 | 1.20× |
| `w4a8|avx512|t1|16x4096x4096` | mr4 nr4 nb128 kb0 o0 | 99 | 229.6 | 1.17× |
| `w4a8|avx512|t1|1x11008x4096` | mr1 nr1 nb128 kb0 o0 | 46 | 108.5 | 1.07× |
| `w4a8|avx512|t1|1x4096x11008` | mr1 nr1 nb128 kb0 o0 | 46 | 103.3 | 1.12× |
| `w4a8|avx512|t1|1x4096x14336` | mr1 nr2 nb32 kb0 o0 | 46 | 106.5 | 1.09× |
| `w4a8|avx512|t1|1x4096x4096` | mr1 nr1 nb32 kb0 o0 | 46 | 91.8 | 1.08× |
| `w4a8|avx512|t1|64x11008x4096` | mr4 nr4 nb256 kb0 o1 | 99 | 235.4 | 1.16× |
| `w4a8|avx512|t1|64x4096x4096` | mr4 nr4 nb128 kb0 o1 | 99 | 269.9 | 1.20× |
| `w4a8|avx512|t1|8x4096x14336` | mr4 nr4 nb128 kb0 o1 | 99 | 249.2 | 1.23× |
| `w4a8|avx512|t4|16x11008x4096` | mr4 nr4 nb64 kb0 o1 | 99 | 870.7 | 1.21× |
| `w4a8|avx512|t4|16x4096x4096` | mr8 nr2 nb128 kb0 o0 | 99 | 756.2 | 1.06× |
| `w4a8|avx512|t4|1x11008x4096` | mr1 nr1 nb128 kb0 o0 | 46 | 268.1 | 1.04× |
| `w4a8|avx512|t4|1x4096x11008` | mr1 nr4 nb128 kb0 o0 | 46 | 334.0 | 1.07× |
| `w4a8|avx512|t4|1x4096x14336` | mr1 nr1 nb64 kb0 o0 | 46 | 355.4 | 1.05× |
| `w4a8|avx512|t4|1x4096x4096` | mr1 nr1 nb128 kb0 o0 | 46 | 268.8 | 1.06× |
| `w4a8|avx512|t4|64x11008x4096` | mr4 nr4 nb32 kb0 o0 | 99 | 916.8 | 1.10× |
| `w4a8|avx512|t4|64x4096x4096` | mr4 nr4 nb64 kb0 o0 | 99 | 851.4 | 1.23× |
| `w4a8|avx512|t4|8x4096x14336` | mr4 nr4 nb64 kb0 o0 | 99 | 783.2 | 1.23× |
