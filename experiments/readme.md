# Numerical Experiments

This directory contains scripts used to reproduce the quantitative results in the manuscript:
**"A Message-passing Perspective on Ptychographic Phase Retrieval"**.

Each script corresponds to a specific figure or table in the paper, and runs simulations under controlled settings using synthetic data.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment](#environment)
3. [Available Experiments](#available-experiments)
4. [How to Run](#how-to-run)

---

## Overview

The scripts in this directory benchmark the performance of **Ptycho-EP** and conventional ptychographic solvers (PIE, ePIE, etc.) across various conditions.

---

## Environment

All experiments assume the following baseline setup unless otherwise specified:

- **Object size**: 512 × 512 pixels
- **Probe size**: 64 × 64 or 128 × 128 pixels
- **Noise**: Gaussian or Poisson noise (SNR = 30 dB)

Please install the required packages from the root `requirements.txt`.

---

## Available Experiments

### 1. `run_reconstruction_vs_scan_density.py`
- **Purpose**: Reproduces *Table I* in the manuscript.
- **What it tests**: Object reconstruction performance (PMSE in dB) as a function of the sampling ratio.
- **Methods compared**: PIE vs Ptycho-EP (gaussian or sparse prior)
- **Outputs**:
  - Median PMSE per method
  - Convergence curves (saved as `results/compare_pie_ep/convergence_curve.png`)

### 2. (planned) `run_blind_reconstruction_vs_overlap.py`
- **Purpose**: Will reproduce *Figure 11* (overlap vs fitness convergence).
- **What it will test**: Performance under limited overlap in blind ptychography.

---

## How to Run

Each script supports command-line arguments. Below are typical usage examples:

### PIE vs Ptycho-EP (Table I benchmark)

```bash
python run_reconstruction_vs_scan_density.py \
    --step 17.0 \
    --noise 3.4 \
    --trials 10 \
    --object lily \
    --prior gaussian
```

## Reproducing Table I (Sampling Ratio vs PMSE)

To reproduce Table I, use the script `run_reconstruction_vs_scan_density.py` with various scan step sizes (which control sampling ratio α). See [`table1_results.md`](table1_results.md) for full command-line usage and PMSE values.

