
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
  - `lily`: sparser image
  - `cameraman`: dense image
- **Noise level**:
  - To ensure a consistent **SNR ≈ 30.0 ± 0.1 dB**, the noise variance was slightly adjusted per scan step size.
  - `lily`: variance = 3.3–3.4e-5
  - `cameraman`: variance = 11.5–12.2e-5
- **PMSE**: Computed over central 256×256 region
- **Sparse prior** indicates use of Bernoulli–Gaussian prior in Ptycho-EP (`--prior sparse`)

---

## Results Summary (with IQR)

### α = 4.0 (step = 16.93)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | 22.98 (19.93–24.86)     | 29.48 (29.47–29.49)         |
| lily      | Sparse   | –                       | 34.07 (34.06–34.09)         |
| cameraman | Gaussian | 28.76 (26.27–28.81)     | 29.80 (28.48–29.85)         |

---

### α = 3.0 (step = 19.78)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | 0.36 (–0.07–0.56)       | 26.66 (26.64–26.68)         |
| lily      | Sparse   | –                       | 32.22 (32.20–32.23)         |
| cameraman | Gaussian | 19.58 (18.95–23.84)     | 26.49 (24.96–26.73)         |

---

### α = 2.5 (step = 21.90)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | –0.35 (–0.73––0.25)     | 22.62 (22.45–22.76)         |
| lily      | Sparse   | –                       | 30.86 (30.86–30.88)         |
| cameraman | Gaussian | 11.04 (9.99–13.10)      | 20.09 (19.24–21.23)         |

---

### α = 2.4 (step = 22.45)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | –0.91 (–1.63––0.62)     | 20.77 (20.11–20.87)         |
| lily      | Sparse   | –                       | 30.54 (30.52–30.56)         |
| cameraman | Gaussian | 9.77 (7.69–10.92)       | 19.38 (18.17–19.65)         |

---

### α = 2.3 (step = 22.95)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | –1.01 (–1.56––0.77)     | 16.60 (15.63–16.85)         |
| lily      | Sparse   | –                       | 27.23 (24.16–29.09)         |
| cameraman | Gaussian | 8.67 (6.74–9.76)        | 15.26 (14.79–17.10)         |

---

### α = 2.2 (step = 23.60)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | –1.41 (–2.10––0.89)     | 1.82 (1.76–1.91)            |
| lily      | Sparse   | –                       | 8.18 (7.54–10.32)           |
| cameraman | Gaussian | 6.62 (4.58–8.55)        | 14.13 (13.03–14.93)         |

---

### α = 2.1 (step = 24.20)

| Object    | Prior    | PIE PMSE [dB]           | Ptycho-EP PMSE [dB]        |
|-----------|----------|-------------------------|-----------------------------|
| lily      | Gaussian | –1.68 (–2.21––1.13)     | 0.00 (0.00–0.00)            |
| lily      | Sparse   | –                       | 2.21 (0.94–2.60)            |
| cameraman | Gaussian | 3.43 (2.43–3.82)        | 12.86 (11.85–13.84)         |

---

## Remarks

- **Ptycho-EP consistently outperforms PIE**, especially at low α (2.1–2.4), where PIE often fails catastrophically (PMSE < 0 dB).
- **Sparse prior** significantly improves reconstruction under limited sampling, confirming the effect described in the manuscript.
