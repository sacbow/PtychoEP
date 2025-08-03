from typing import List, Tuple
from ..backend import np
from .data import DiffractionData
from .core import Ptycho

def generate_diffraction(ptycho: Ptycho, positions: List[Tuple[int, int]]) -> List[DiffractionData]:
    """
    タイコグラフィのforwardモデルに基づき回折像データ（ノイズなし）を生成する。

    Args:
        ptycho: オブジェクトとプローブが設定されたPtychoインスタンス
        positions: スキャン位置リスト [(y, x), ...]

    Returns:
        DiffractionDataオブジェクトのリスト（ノイズなし）
    """
    if ptycho.obj is None or ptycho.prb is None:
        raise ValueError("Ptycho object must have both obj and probe set before forward generation.")
    if ptycho.obj_len is None or ptycho.prb_len is None:
        raise ValueError("Object and probe dimensions are not initialized.")

    prb_len = ptycho.prb_len
    diffs: List[DiffractionData] = []
    fft2 = np().fft.fft2
    for pos in positions:
        y, x = pos
        # オブジェクトパッチを抽出
        obj_patch = ptycho.obj[
            y - prb_len // 2 : y + prb_len // 2,
            x - prb_len // 2 : x + prb_len // 2
        ]
        # 回折像（振幅）
        diff = np().abs(fft2(obj_patch * ptycho.prb, norm = "ortho"))
        #indices
        yy = np().arange(y - prb_len // 2, y + prb_len // 2)
        xx = np().arange(x - prb_len // 2, x + prb_len // 2)
        Y, X = np().meshgrid(yy, xx, indexing='ij')
        diffs.append(DiffractionData(position=pos, diffraction=diff, indices=(Y, X)))
    return diffs
