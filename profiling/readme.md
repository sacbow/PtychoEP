# Profiling: Ptychographic Algorithms

This directory contains profiling and benchmarking scripts for ptychographic reconstruction algorithms implemented in this repository.

---

## Table of Contents

1. [Overview](#overview)
2. [Profiling Setup](#profiling-setup)
3. [Fixed-Probe Algorithms](#fixed-probe-algorithms)
    - PIE
    - PtychoEP (without probe update)
4. [Probe-Update Algorithms](#probe-update-algorithms)
    - ePIE
    - rPIE
    - DM
    - PtychoEP (with EM-based probe update)
5. [Execution Tips](#execution-tips)
6. [Performance Notes](#performance-notes)

---

## Overview

Each script in this directory allows you to either measure total execution time or run a detailed line-by-line profiler using Python's built-in `cProfile`.

Two modes are supported:

- **Timing mode** (default): Uses `time.perf_counter()` or `cupy.cuda.Event()` to measure execution time
- **Profiling mode** (`--profile`): Uses `cProfile` with configurable output

---

## Profiling Setup

All profiling scripts support the following options:

```bash
--backend {numpy,cupy}      # CPU vs GPU
--niter <int>               # Number of iterations
--num_points <int>          # Number of scan positions
--use_noise                 # Enable Gaussian noise (variance = 1e-3)
--profile                   # Enable cProfile
--profile_output <file>     # Save profile result to file
```

### Example:
python profiling/profile_pie.py --backend numpy --niter 50 --profile

### Benchmark Environment of the developer:

- **OS:** Windows 11 Home, ver. 24H2, OS build 26100 4652
- **CPU:** Intel Core i7-14650K (16 cores, 24 threads)  
- **RAM:** 32 GB 
- **GPU:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)  
- **NVIDIA Driver:** 576.02
- **CUDA Toolkit:** 12.9
- **Python:** 3.10.5 (venv)  
- **Libraries:**
  - NumPy: 2.2.6
  - CuPy: 13.5.1

- **Note**: Results are device-dependent and may vary on different hardware or driver configurations.

### Profiling Configuration

| Parameter      | Value     |
|----------------|-----------|
| Iterations     | 50        |
| Scan points    | 200       |
| Probe size     | 128×128   |
| Image size     | 512×512   |
| Noise          | None      |

---
## Fixed-Probe Algorithms

These algorithms assume that the illumination function of the probe is fixed and known.

### PIE

  - Script: `profiling/profile_pie.py`
  - Processes one scan position at a time
  - FFT-dominant but memory-bound

#### Execution Time (50 iterations)

| Backend | Execution Time [sec] | Notes |
|---------|----------------------|-------|
| NumPy   | 4.6                  | `_pocketfft` dominates time (CPU FFT) |
| CuPy    | 3.8                  | No significant acceleration due to unhidden GPU launch overhead |

#### Script Used

```bash
python profiling/profile_pie.py --backend numpy --niter 50
python profiling/profile_pie.py --backend cupy --niter 50
```

### PtychoEP (without EM-based probe update)

  - Script: `profiling/profile_ep.py`
  - Expectation Propagation–based phase retrieval algorithm
  - Probe is fixed (EM update disabled)
  - Performs message passing and belief update scan-by-scan

#### Execution Time (50 iterations)

| Backend | Execution Time [sec] | Notes |
|---------|----------------------|-------|
| NumPy   | 7.9                  | ≈35–40% FFT, ≈30% message passing, ≈30% belief updates |
| CuPy    | 11.8                  | Significant overhead in message passing; GPU under-utilized |

#### Script Used

```bash
python profiling/profile_ep.py --backend numpy --niter 50
python profiling/profile_ep.py --backend cupy --niter 50
```

#### Key Observations
 - The runtime is approximately 1.7× that of PIE, which reflects the cost of uncertainty-aware message passing and belief estimation.
 - FFT operations are still the dominant component on CPU, but account for only ~38% of total time.
 - The remaining cost is split between:
    - Laplace-approximate update of the posterior in Likelihood module
    - UncertainArray operations: including precision fusion (damp_with) and division (__truediv__)
 - On GPU, elementwise operations and scatter-like memory updates become performance bottlenecks due to lack of batching and kernel fusion.



---
## Probe-Update Algorithms

These algorithms jointly recover probe and object from the diffraction dataset.

### ePIE

  - Script: `profiling/profile_epie.py`
  - Processes scan points sequentially.

#### Execution Time (50 iterations)

| Backend | Execution Time [sec] | Notes |
|---------|----------------------|-------|
| NumPy   | 5.5                  | FFT dominates (~52%); update_object is nontrivial due to scatter-style access |
| CuPy    | 6.2                  | FFT is faster, but object/probe updates become bottlenecks|

#### Script Used

```bash
python profiling/profile_epie.py --backend numpy --niter 50
python profiling/profile_epie.py --backend cupy --niter 50
```

### rPIE

  - Script: `profiling/profile_rpie.py`
  - rPIE Stabilizes object and probe updates using relaxation.
  - Processes scan points sequentially.

#### Execution Time (50 iterations)

| Backend | Execution Time [sec] | Notes |
|---------|----------------------|-------|
| NumPy   | 5.6                  | Similar to ePIE; extra weighting cost is minor |
| CuPy    | 6.3                  | FFT is faster; update steps dominate due to element-wise max operations|

#### Script Used

```bash
python profiling/profile_rpie.py --backend numpy --niter 50
python profiling/profile_rpie.py --backend cupy --niter 50
```

### Difference Map

  - Script: `profiling/profile_dm.py`
  - Uses batch processing across all scan positions per iteration.
  - Well-suited for GPU execution due to contiguous memory operations.

#### Execution Time (50 iterations)

| Backend | Execution Time [sec] | Notes |
|---------|----------------------|-------|
| NumPy   | 11.3                 | Dominated by ufunc.at (scatter-style object updates) |
| CuPy    | 1.3                  | Massive acceleration due to full batch processing; FFT and elementwise operations hidden|

#### Remarks
- On GPU, the remaining bottleneck is the scatter-style object update, not FFT.

#### Script Used

```bash
python profiling/profile_dm.py --backend numpy --niter 50
python profiling/profile_dm.py --backend cupy --niter 50
```

### PtychoEP (with EM-based Probe Update)

  - Script: `profiling/profile_ep.py`
  - The variant of Ptycho-EP profiled here includes EM-based probe update, executed once every iteration.
  - Unlike Difference Map, this algorithm operates scan-by-scan, not in batch mode.

#### Execution Time (50 iterations)

| Backend | Execution Time [sec] | Notes |
|---------|----------------------|-------|
| NumPy   | 10.7                 | Dominated by FFT and EM-based probe updates; backward pass in likelihood is significant |
| CuPy    | 12.8                 | GPU overhead and kernel launch costs outweigh FFT gains; damp_with and __truediv__ become bottlenecks|

#### Remarks
- EM-based probe update uses stacked UA patches and incurs costs from np.stack and slicing operations.

#### Script Used

```bash
python profiling/profile_ep.py --backend numpy --niter 50 --n_probe_update 1
python profiling/profile_ep.py --backend cupy --niter 50 --n_probe_update 1
```