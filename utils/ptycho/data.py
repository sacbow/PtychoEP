from dataclasses import dataclass, field
from typing import Tuple, Any
from ..backend import np

@dataclass
class DiffractionData:
    """
    1スキャン位置に対応する回折像データを格納するクラス。

    Attributes:
        position (Tuple[int, int]): スキャン位置 (y, x) ピクセル座標。
        diffraction: 回折像データ (2D array)。
        meta (dict): 任意のメタデータ（例: 時刻、条件ラベル）。
    """
    position: Tuple[int, int]
    diffraction: np().ndarray
    meta: dict = field(default_factory=dict)
    indices: Tuple[np().ndarray, np().ndarray] = None

    def intensity(self) -> np().ndarray:
        """回折像の強度（振幅^2）を返す。"""
        return np().abs(self.diffraction) ** 2

    def show(self, ax=None, log_scale=True, cmap="jet"):
        """matplotlibを用いて回折像を可視化。"""
        import matplotlib.pyplot as plt
        from ..backend import np

        data = self.diffraction
        if log_scale:
            data = np().log10(np().abs(data) + 1e-8)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(data, cmap=cmap)
        ax.set_title(f"Pos: {self.position}")
        ax.axis("off")
        return ax

    def summary(self) -> str:
        """データの簡単な情報を文字列で返す。"""
        return f"Pos={self.position}, shape={self.diffraction.shape}, meta={self.meta}"
