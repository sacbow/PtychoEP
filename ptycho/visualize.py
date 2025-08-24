import matplotlib.pyplot as plt
from ptychoep.backend.backend import np
import numpy as _np

def to_numpy(arr):
    """
    Convert to NumPy array if input is a CuPy array; otherwise return as-is.

    Args:
        arr: Input array (NumPy or CuPy)

    Returns:
        Converted NumPy array or original input
    """
    import cupy
    if isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return arr


def compute_illumination(ptycho):
    """
    Compute illumination map and sampling ratio.

    This function uses the precomputed patch indices from forward propagation
    to calculate the cumulative illumination distribution and the effective
    sampling rate (alpha) over the object domain.

    Args:
        ptycho: A Ptycho object with defined scan and probe

    Returns:
        scan_img: 2D array of accumulated illumination
        alpha: Sampling rate defined as (J * probe_area) / illuminated_area
    """
    prb_sq = np().abs(ptycho.prb) ** 2
    scan_img = np().zeros((ptycho.obj_len, ptycho.obj_len), dtype=float)

    for d in ptycho._diff_data:
        Y, X = d.indices
        np().add.at(scan_img, (Y, X), prb_sq)

    sampling_number = np().sum(scan_img > 0.1 * np().max(scan_img))
    alpha = (ptycho.prb_len ** 2 * len(ptycho.scan_pos)) / sampling_number

    return to_numpy(scan_img), float(alpha)


def show_scan_and_diffs(ptycho, num_patterns=4, log_scale=True, cmap_obj="gray", cmap_diff="jet"):
    """
    Visualize scan illumination and sample diffraction patterns.

    Args:
        ptycho: Ptycho object containing scan metadata and diffraction data
        num_patterns: Number of diffraction patterns to visualize
        log_scale: Whether to apply log scaling to diffraction amplitudes
        cmap_obj: Colormap for scan image
        cmap_diff: Colormap for diffraction images

    Returns:
        Matplotlib Figure object
    """
    from numpy.fft import fftshift

    scan_img, alpha = compute_illumination(ptycho)
    fig, axes = plt.subplots(1, num_patterns + 1, figsize=(3 * (num_patterns + 1), 3))

    # Scan illumination map
    axes[0].imshow(scan_img, cmap=cmap_obj)
    axes[0].axis("off")
    axes[0].set_title(f"Scanned positions\n(alpha={alpha:.2f})", fontsize=12)

    # Diffraction patterns
    for i in range(num_patterns):
        ax = axes[i + 1]
        ax.axis("off")
        diff = ptycho.diffs[i]
        diff = to_numpy(diff)
        if log_scale:
            diff = _np.log10(_np.abs(fftshift(diff)) + 1e-8)
        else:
            diff = _np.abs(fftshift(diff))
        ax.imshow(diff, cmap=cmap_diff)
        ax.set_title(f"Diffraction {i+1}", fontsize=12)

    plt.tight_layout()
    plt.show()
    return fig
