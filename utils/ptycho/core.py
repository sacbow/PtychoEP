from typing import List, Tuple
from .data import DiffractionData
from ..backend import np


class Ptycho:
    """
    タイコグラフィデータを管理するクラス。
    旧PtychoクラスのAPI互換性を保ちつつ、内部的にはDiffractionDataを用いる。
    """

    def __init__(self, object: np().ndarray, probe: np().ndarray, scan_positions: List[Tuple[int, int]], noise: float = 0.0):
        """
        Args:
            object: 物体の複素振幅分布 (2D array)
            probe: プローブ関数 (2D array)
            scan_positions: スキャン座標リスト [(y, x), ...]
            noise: ノイズ標準偏差
        """
        self.obj = object
        self.prb = probe
        self.obj_len = object.shape[0]
        self.prb_len = probe.shape[0]
        self.noise = noise

        # DiffractionDataのリストを生成
        self._diff_data: List[DiffractionData] = generate_diffraction(object, probe, scan_positions, noise)

        # 照明マップ・サンプリング率を計算
        self.scan_img, self.sampling_ratio = self._compute_illumination()

    # --- 旧API互換: scan_pos/diffs ---
    @property
    def scan_pos(self) -> List[Tuple[int, int]]:
        """スキャン座標リスト (旧API互換)"""
        return [d.position for d in self._diff_data]

    @property
    def diffs(self) -> List[np().ndarray]:
        """回折像リスト (旧API互換)"""
        return [d.diffraction for d in self._diff_data]

    # --- 照明マップ生成 ---
    def _compute_illumination(self):
        prb_abs = np().abs(self.prb)
        scan_img = np().zeros((self.obj_len, self.obj_len), dtype=float)
        for d in self._diff_data:
            y, x = d.position
            scan_img[y - self.prb_len // 2: y + self.prb_len // 2,
                     x - self.prb_len // 2: x + self.prb_len // 2] += prb_abs ** 2
        sampling_number = np().sum(scan_img > 0.1 * np().max(scan_img))
        alpha = (self.prb_len ** 2 * len(self._diff_data)) / sampling_number
        return scan_img, alpha

    # --- 可視化 (旧show相当) ---
    def show(self, L: int = 4):
        """スキャン位置と代表的な回折像を可視化"""
        import matplotlib.pyplot as plt
        from ..backend import np
        from numpy.fft import fftshift

        fig, ax = plt.subplots(1, L + 1, figsize=(3 * (L + 1), 3))
        ax[0].axis("off")
        ax[0].imshow(self.scan_img, cmap="gray")
        ax[0].set_title("Scanned positions", fontsize=15)

        for l in range(1, L + 1):
            ax[l].axis("off")
            diff = self._diff_data[l - 1].diffraction
            ax[l].imshow(np().log10(fftshift(diff)), cmap="jet")
            ax[l].set_title(f"Diffraction {l}", fontsize=15)

        plt.show()
