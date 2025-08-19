import math
from typing import List, Tuple
from backend.backend import np
from rng.rng_utils import get_rng, randint

def generate_spiral_scan_positions(
    image_size: int,
    probe_size: int,
    num_points: int,
    step: float = 5,
    center: Tuple[int, int] = None,
    bounds_check: bool = True
) -> List[Tuple[int, int]]:
    """
    らせん状スキャン座標を生成する。

    Args:
        image_size: 画像サイズ（正方形を前提）
        probe_size: プローブサイズ（正方形を前提）
        num_points: 生成するスキャン点数
        step: らせんの広がり係数
        center: スキャン中心座標（Noneなら画像中心）
        bounds_check: プローブが画像からはみ出さないよう制限するか

    Returns:
        スキャン位置のリスト [(y,x), ...]
    """
    if center is None:
        center = (image_size // 2, image_size // 2)

    positions: List[Tuple[int, int]] = []
    for i in range(num_points):
        x = int(center[1] + step * math.sqrt(i) * math.cos(i * 2.399967))
        y = int(center[0] + step * math.sqrt(i) * math.sin(i * 2.399967))
        if bounds_check:
            if (x - probe_size // 2 < 0 or x + probe_size // 2 > image_size or
                y - probe_size // 2 < 0 or y + probe_size // 2 > image_size):
                continue
        positions.append((y, x))
    return positions

def generate_grid_scan_positions(
    image_size: int,
    probe_size: int,
    step: int = 16,
    jitter: int = 0,  # ずらしの範囲（例: jitter=3なら±3の範囲でランダム移動）
    bounds_check: bool = True
) -> List[Tuple[int, int]]:
    """
    格子状スキャン座標を生成する（オプションでランダムジッターを追加）。

    Args:
        image_size: 画像サイズ（正方形を前提）
        probe_size: プローブサイズ（正方形を前提）
        step: 格子間隔（ピクセル単位）
        jitter: 格子位置を±jitter範囲でランダムにずらす（0ならオフ）
        bounds_check: プローブが画像からはみ出さないよう制限するか

    Returns:
        スキャン位置のリスト [(y, x), ...]
    """
    positions: List[Tuple[int, int]] = []
    rng = get_rng()  # backend対応の乱数生成器

    for y in range(0, image_size, step):
        for x in range(0, image_size, step):
            y_pos = y
            x_pos = x
            if jitter > 0:
                y_pos += randint(rng, -jitter, jitter + 1)
                x_pos += randint(rng, -jitter, jitter + 1)
            # 境界チェック
            if bounds_check:
                if (x_pos - probe_size // 2 < 0 or x_pos + probe_size // 2 > image_size or
                    y_pos - probe_size // 2 < 0 or y_pos + probe_size // 2 > image_size):
                    continue
            positions.append((y_pos, x_pos))

    return positions