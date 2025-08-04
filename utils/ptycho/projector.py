from ..backend import np

def Fourier_projector(exit_wave, target_amp, eps: float = 1e-7):
    """
    Fourierドメインで振幅制約を課す投影演算と誤差計算を同時に実行。

    Args:
        exit_wave (ndarray): exit wave (複素振幅)
        target_amp (ndarray): 観測振幅（sqrt(I)）
        eps (float): 正規化時のゼロ除算回避用微小値
    
    Returns:
        tuple:
            proj_wave (ndarray): 投影後のexit wave
            error (float): 観測振幅とのMSE誤差
    """

    fft2 = np().fft.fft2
    ifft2 = np().fft.ifft2

    # FFT計算
    freq_wave = fft2(exit_wave, norm="ortho")

    # 振幅と誤差
    pred_amp = np().abs(freq_wave)
    error = np().mean((target_amp - pred_amp) ** 2)

    # 振幅置換（位相は維持）
    projected_freq = target_amp * freq_wave / (pred_amp + eps)
    proj_wave = ifft2(projected_freq, norm="ortho")

    return proj_wave , float(error)
