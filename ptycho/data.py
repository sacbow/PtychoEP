from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
from backend.backend import np

IdxType = Optional[Tuple[slice, slice]]

@dataclass
class DiffractionData:
    position: Tuple[int, int]                     # (y, x)
    diffraction: np().ndarray
    meta: dict = field(default_factory=dict)
    indices: IdxType = None                       # ← ndarray から slice に変更
    gamma_w: Optional[float] = None

    def intensity(self) -> np().ndarray:
        """回折像の強度（振幅^2）を返す。"""
        return np().abs(self.diffraction) ** 2
    
    def get_gamma_w(self):
        if self.gamma_w is None:
            raise ValueError("gamma_w is not set for this diffraction data.")
        return self.gamma_w


    def show(self, ax=None, log_scale=True, cmap="jet"):
        """matplotlibを用いて回折像を可視化。"""
        import matplotlib.pyplot as plt
        from backend.backend import np

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
    
    def __hash__(self):
        # 基本的にはインスタンスIDに基づくハッシュ（ユニーク性を担保）
        return id(self)

    def __eq__(self, other):
        return self is other
