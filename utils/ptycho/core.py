from typing import List, Tuple, Optional
from .data import DiffractionData
from ..backend import np

class Ptycho:
    """
    タイコグラフィデータを管理するクラス。
    DiffractionDataオブジェクトのリスト管理と旧API互換性（scan_pos, diffs）を提供する。
    """

    def __init__(self):
        # オブジェクト・プローブ・スキャン座標・回折データは後から設定可能
        self.obj: Optional[np().ndarray] = None
        self.prb: Optional[np().ndarray] = None
        self.obj_len: Optional[int] = None
        self.prb_len: Optional[int] = None
        self._diff_data: List[DiffractionData] = []

    # --- オブジェクト・プローブ設定 ---
    def set_object(self, obj: np().ndarray):
        """物体（複素振幅分布）を設定"""
        if obj.shape[0] != obj.shape[1]:
            raise ValueError("shape of object should be square")
        self.obj = obj
        self.obj_len = obj.shape[0]

    def set_probe(self, probe: np().ndarray):
        """プローブ関数を設定"""
        if probe.shape[0] != probe.shape[1]:
            raise ValueError("shape of probe should be square")
        self.prb = probe
        self.prb_len = probe.shape[0]

    # --- DiffractionData管理 ---
    def add_diffraction_data(self, diff_data: DiffractionData):
        """1つのDiffractionDataを追加"""
        if not isinstance(diff_data, DiffractionData):
            raise TypeError("diff_data must be a DiffractionData instance")
        self._diff_data.append(diff_data)

    def add_diffraction_data_list(self, diff_data_list: List[DiffractionData]):
        """複数のDiffractionDataを追加"""
        for d in diff_data_list:
            self.add_diffraction_data(d)

    def clear_diffraction_data(self):
        """登録されている回折データを全て削除"""
        self._diff_data.clear()

    def set_diffraction_from_forward(self, diff_list: List[DiffractionData], append: bool = False):
        """
        forward計算で生成した回折データを登録する。

        Args:
            diff_list: DiffractionDataのリスト（generate_diffractionの結果）
            append: Trueなら既存データに追加、Falseならクリアして置換
        """
        if not append:
            self.clear_diffraction_data()
        self.add_diffraction_data_list(diff_list)

    def forward_and_set_diffraction(self, positions: List[Tuple[int, int]], append: bool = False):
        """
        与えられたスキャン位置に基づいてforward計算を実行し、
        結果の回折データを登録する。

        Args:
            positions: スキャン位置リスト [(y,x), ...]
            append: Trueなら既存データに追加、Falseならクリアして置換
        """
        from .forward import generate_diffraction
        diff_list = generate_diffraction(self, positions)
        self.set_diffraction_from_forward(diff_list, append=append)

    # --- 旧API互換プロパティ ---
    @property
    def scan_pos(self) -> List[Tuple[int, int]]:
        """スキャン座標リスト (旧API互換)"""
        return [d.position for d in self._diff_data]

    @property
    def diffs(self) -> List[np().ndarray]:
        """回折像リスト (旧API互換)"""
        return [d.diffraction for d in self._diff_data]
