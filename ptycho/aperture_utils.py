from ptychoep.backend.backend import np

def circular_aperture(size: int, r: float, dtype=None):
    """
    円形アパーチャ（プローブ）を生成する。

    Args:
        size (int): 生成する配列の縦横サイズ (n×n)。
        r (float): アパーチャの半径を [0, 0.5] の範囲で指定 (画像サイズに対する相対値)。
        dtype: 出力配列のdtype（省略時はbackend依存のfloat32）。

    Returns:
        xp.ndarray: 円形アパーチャ配列 (size×size)。
    """
    xp = np()
    if dtype is None:
        dtype = xp.complex64

    if not (0.0 < r <= 0.5):
        raise ValueError(f"r must be in (0, 0.5], got {r}")

    # 座標系 (中心を原点に)
    y, x = xp.ogrid[-size//2:size//2, -size//2:size//2]
    radius = xp.sqrt(x**2 + y**2)

    # 半径 r*size の円形マスク
    aperture = xp.zeros((size, size), dtype=dtype)
    aperture[radius <= (r * size)] = 1.0
    return aperture
