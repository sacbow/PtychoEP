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
ptychoep/
├── ptycho/                             # Container for ptychographic datasets
├── backend/                            # backend abstraction (numpy/cupy)
├── utils/                              # Utilities (io)
├── rng/                                # backend abstraction of random number generator
├── ptychoep/
│   ├── object.py                       # Variable node representing the object
│   ├── probe.py                        # Factor node representing the multiplication with probe
│   ├── fft_channel.py                  # Factor node representing the fourier transform
│   ├── likelihood.py                   # Factor node representing the likelihood
│   ├── prior.py                        # Factor node representing the prior
│   ├── uncertain_array.py              # Abstraction of gaussian distribution
|   ├── accumulative_uncertain_array    # Data structure used in the object node
├── profiling/                          # Profiling and benchmarking scripts
├── experiments/                        # scripts for numerical experiments
└── README.md

```

## Quick start
```bash
# Clone and set up virtual environment
git clone https://github.com/yourname/PtychoEP.git
cd ptychoep
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Example: Profile PtychoEP with EM algorithm
```bash
python profiling/profile_ep.py --backend numpy --niter 50 --n_probe_update 1
```

### Example: Run numerical experiments on ptychographic reconstruction with sparse prior
```bash
python experiments/script/run_blind_reconstruction.py --object lily --step 16.93 --noise 3.4 --trials 10 --prior sparse
```


## Contact
For questions, please open an issue or contact:
- Hajime Ueda (ueda@mns.k.u-tokyo.ac.jp)