from ..backend import np

def Fourier_projector(exit_wave, target_amp, eps: float = 1e-7, return_per_scan: bool = False):
    """
    Fourierドメインで振幅制約を課す投影演算と誤差計算（バッチ対応版）。

    Args:
        exit_wave (ndarray): exit wave (複素振幅), shape=(n_scan, H, W) or (H, W)
        target_amp (ndarray): 観測振幅 sqrt(I), shape=(n_scan, H, W) or (H, W)
        eps (float): ゼロ除算回避用微小値
        return_per_scan (bool): Trueならスキャンごとの誤差配列を返す

    Returns:
        proj_wave (ndarray): 投影後exit wave (shapeは入力に対応)
        error (float or ndarray): 誤差（スカラーorスキャンごと1D配列）
    """
    xp = np()
    fft2, ifft2 = xp.fft.fft2, xp.fft.ifft2

    # FFT (バッチ対応: xp.fft.fft2は先頭次元をバッチ扱い)
    freq_wave = fft2(exit_wave, norm="ortho")

    # 振幅と誤差
    pred_amp = xp.abs(freq_wave)
    sq_error = (target_amp - pred_amp) ** 2

    if return_per_scan and exit_wave.ndim == 3:
        error = sq_error.reshape(len(sq_error), -1).mean(axis=1)  # スキャン単位MSE
    else:
        error = xp.mean(sq_error)

    # 振幅置換（位相保持）
    projected_freq = target_amp * freq_wave / (pred_amp + eps)
    proj_wave = ifft2(projected_freq, norm="ortho")

    return proj_wave, (error if return_per_scan else float(error))
