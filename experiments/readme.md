# README: Numerical Experiments for PtychoEP

This directory contains scripts for the demonstration of Ptycho-EP. These experiments compare the reconstruction performance of the proposed method (Ptycho-EP) against conventional algorithms.

---

## Contents

### 1. `run_reconstruction_vs_scan_density.py`

Performs reconstruction using PIE and Ptycho-EP on synthetic diffraction data with different noise levels and scan step sizes.
This experiment assumes a round aperture probe function.

* **Arguments**:

  * `--n_repeats`: number of repeated trials (default: 10)
  * `--noise_type`: `gaussian` or `poisson`
  * `--noise_param`: variance (for Gaussian) or scale (for Poisson)
  * `--step`: scan step size (in pixels)
  * `--num_scans`: number of scans along each axis (default: 11)

* **Example Command**
```bash
  python script/run_reconstruction_vs_scan_density.py  --n_repeats 10 --noise_type poisson --noise_param 3000  --step 12  --num_scans 16
```

* **Output**:

* Reconstruction image (only for the first trial) per method:

```
    result/recon\_EP\_seed\_0.png
    result/recon\_ePIE\_seed\_0.png
```
*  Median convergence curve across trials:
```
    result/convergence_median.png
```

### 2. `run_blind_reconstruction.py`

Performs blind reconstruction using four algorithms (Ptycho-EP, PIE, ePIE, rPIE, DM) from random object initializations. Used to evaluate PMSE and convergence on different object datasets.

* **Arguments**:

  * `--step`: scan step size (default: 16.93)
  * `--noise`: Gaussian noise variance in units of 1e-5 (default: 3.4)
  * `--trials`: number of repeated trials (default: 10)
  * `--object`: `lily` or `cameraman`
  * `--prior`: `gaussian` or `sparse`

* **Example Command**:

```bash
  python script/run_blind_reconstruction.py --n_repeats 10 --noise_type poisson --noise_param 3000 --step 18 --num_scans 11
```

* **Output**:

* Cropped reconstruction images (only for the first trial):

```
    result/recon\_pie\_object\_{name}*step*{step}.png
    result/recon\_ep\_object\_{name}*step*{step}.png
```

* PMSE convergence curve:
```
    result/convergence_object_{name}_step_{step}.png
```

---

## Notes

* All experiments are run with `backend = numpy`.
* The `result/` directory will be created automatically if not present.
* File names include key parameters (`step`, `object`) to avoid overwriting.
* For Poisson noise, photon count is computed.
* Median convergence curves are plotted on a log scale.

---
### 3. `run_uncertainty_validation.py`
Evaluates the pixel-wise uncertainty estimates produced by Ptycho-EP.
This script performs a reconstruction on synthetic Gaussian-noise data, computes the normalized reconstruction error with respect to the posterior standard deviation, and visualizes the error distribution.

## Description
This script runs a single reconstruction using PtychoEP and evaluates the uncertainty map (i.e., the posterior precision Γ_{O,int}) by checking how the reconstruction error aligns with the estimated standard deviation.

## Output
All output files are saved to the experiment/result/ directory
```
  experiment/result/reconstruction_amplitude.png
  experiment/result/estimated_stddev_with_colorbar_log_fixed.png
  experiment/result/uncertainty_histogram.png
```

- reconstruction_amplitude.png: Aligned amplitude of the reconstructed object.
- estimated_stddev_with_colorbar_log_fixed.png: Estimated standard deviation map with logarithmic color scale (1e0–1e-2), and colorbar.
- uncertainty_histogram.png: Histogram of the normalized error |x̂ − x| / σ, showing the distribution across the central 256×256 region.

The script also prints the percentage of pixels within the range [0.5σ, 2.0σ], providing empirical validation of the uncertainty estimates.

* **Example Command**
```bash
  python script/run_uncertainty_validation.py
```