# PtychoEP: Expectation Propagation for Ptychographic Phase Retrieval

PtychoEP is a modular implementation of ptychographic phase retrieval algorithms based on Expectation Propagation (EP), along with standard baseline methods such as PIE, ePIE, rPIE, and Difference Map.

This repository is designed for both research reproducibility and extensibility toward advanced Bayesian inference in coherent imaging.

## Features
 - Modular implementation of PtychoEP
 - Support for fixed-probe and probe-update workflows (EM-based estimation)
 - Backends supported: numpy (CPU), cupy (GPU)
 - Benchmark-ready profiling scripts for all algorithms
 - Clean abstraction of
    - Forward models
    - Uncertainty-aware updates (UncertainArray)
    - Scan pattern utilities (e.g. spiral scanning)
    - Prior/Likelihood modules

## Repository Structure
```
PtychoEP/
├── core/                       # Core algorithm drivers
├── utils/                      # Utilities (scan gen, aperture, RNG, etc.)
├── modules/
│   ├── fft_channel.py          # Fourier forward model
│   ├── likelihood.py           # Likelihood module (|z| + noise)
│   ├── prior.py                # Prior module for object/probe
│   ├── uncertain_array.py      # Main message passing data structure
├── profiling/                  # Profiling and benchmarking scripts
├── notebooks/                  # Example reconstructions and analysis
└── README.md

```

## Quick start
```bash
# Clone and set up virtual environment
git clone https://github.com/yourname/PtychoEP.git
cd PtychoEP
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Example: Run fixed-probe PIE reconstruction
```bash
python profiling/profile_pie.py --backend numpy --niter 50
```

### Example: Run PtychoEP with EM-based probe update
```bash
python profiling/profile_ep.py --backend cupy --niter 50 --n_probe_update 1
```

## Benchmarking & Profiling
Profiling results (CPU/GPU) for all algorithms are documented in [profiling/README.md], including:
- PIE
- ePIE
- rPIE
- Difference Map
- PtychoEP (with/without EM probe update)

All results include runtime breakdown, bottleneck analysis, and hardware info.