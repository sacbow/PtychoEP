
# Table I: Reconstruction Accuracy vs Sampling Ratio (α)

This document summarizes the median PMSE (in dB) for PIE and Ptycho-EP under varying sampling ratios α.  
Each result is computed over **10 trials** with randomized initialization and noise, using the script:

```bash
python run_reconstruction_vs_scan_density.py
```

See `experiments/README.md` for usage details.

---

## Notes

- **Object sets**:
  - `lily`: phase-rich image, moderate sparsity
  - `cameraman`: sparser image
- **Noise level**:
  - To ensure a consistent **SNR ≈ 30.0 ± 0.1 dB**, the noise variance was slightly adjusted per scan step size.
  - `lily`: variance = 3.3–3.4e-5
  - `cameraman`: variance = 11.6–12.1e-5
- **PMSE**: Computed over central 256×256 region
- **Sparse prior** indicates use of Bernoulli–Gaussian prior in Ptycho-EP (`--prior sparse`)

---

## Results Summary

### α = 4.0 (step = 16.93)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | 22.56    | 29.45          |
| lily       | Sparse    |     -    | 34.03          |
| cameraman  | Gaussian  | 28.76    | 29.83          |

---

### α = 3.0 (step = 19.78)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | 0.39     | 26.47          |
| lily       | Sparse    |    -     | 32.09          |
| cameraman  | Gaussian  | 19.49    | 26.51          |

---

### α = 2.5 (step = 21.90)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | -0.35    | 22.56          |
| lily       | Sparse    |    -     | 30.87          |
| cameraman  | Gaussian  | 10.98    | 20.09          |

---

### α = 2.4 (step = 22.45)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | -0.90    | 20.38          |
| lily       | Sparse    |    -     | 30.54          |
| cameraman  | Gaussian  | 9.77     | 18.72          |

---

### α = 2.3 (step = 22.95)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | -1.02    | 16.77          |
| lily       | Sparse    |     -    | 23.33          |
| cameraman  | Gaussian  | 8.67     | 14.82          |

---

### α = 2.2 (step = 23.60)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | -1.41    | 1.94           |
| lily       | Sparse    |     -    | 8.22           |
| cameraman  | Gaussian  | 6.61     | 13.00          |

---

### α = 2.1 (step = 24.20)

| Object     | Prior     | PIE PMSE | Ptycho-EP PMSE |
|------------|-----------|----------|----------------|
| lily       | Gaussian  | -1.68    | 0.00           |
| lily       | Sparse    |     -    | 2.71           |
| cameraman  | Gaussian  | 3.42     | 12.45          |

---

## Remarks

- **Ptycho-EP consistently outperforms PIE**, especially at low α (2.1–2.4), where PIE often fails catastrophically (PMSE < 0 dB).
- **Sparse prior** significantly improves reconstruction under limited sampling, confirming the effect described in the manuscript.
