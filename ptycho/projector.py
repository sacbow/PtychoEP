from ptychoep.backend.backend import np

def Fourier_projector(exit_wave, target_amp, eps: float = 1e-7, return_per_scan: bool = False):
    """
    Enforce amplitude constraints in the Fourier domain via projection, and compute error.

    This function applies a projection operator that replaces the Fourier amplitude of 
    the exit wave with the measured amplitude (target_amp), preserving the phase.
    Optionally returns mean squared error (MSE) per scan.

    Args:
        exit_wave (ndarray): Complex-valued exit wave(s), shape (H, W) or (N, H, W)
        target_amp (ndarray): Measured amplitude sqrt(I), same shape as exit_wave
        eps (float): Small positive number to avoid division by zero
        return_per_scan (bool): If True, returns an array of per-scan projection errors

    Returns:
        proj_wave (ndarray): Projected exit wave with replaced amplitude
        error (float or ndarray): Mean squared error between amplitudes.
            If return_per_scan is True and input is batched (3D), returns per-scan error.
            Otherwise returns scalar average.
    """

    xp = np()
    fft2, ifft2 = xp.fft.fft2, xp.fft.ifft2

    freq_wave = fft2(exit_wave, norm="ortho")

    pred_amp = xp.abs(freq_wave)
    sq_error = (target_amp - pred_amp) ** 2

    if return_per_scan and exit_wave.ndim == 3:
        error = sq_error.reshape(len(sq_error), -1).mean(axis=1)
    else:
        error = xp.mean(sq_error)

    projected_freq = target_amp * freq_wave / (pred_amp + eps)
    proj_wave = ifft2(projected_freq, norm="ortho")

    return proj_wave, (error if return_per_scan else float(error))
