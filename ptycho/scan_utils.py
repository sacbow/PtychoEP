import math
from typing import List, Tuple
from ptychoep.backend.backend import np
from ptychoep.rng.rng_utils import get_rng, randint

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
    jitter: int = 0,  
    bounds_check: bool = True,
    seed: int = None
) -> List[Tuple[int, int]]:
    """
    to be deplicated.
    """
    positions: List[Tuple[int, int]] = []
    rng = get_rng(seed)  # backend対応の乱数生成器

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

def generate_centered_grid_positions(
    image_size: int,
    probe_size: int,
    step: int,
    num_points_y: int,
    num_points_x: int,
    jitter: int = 0,
    seed: int = None,
    bounds_check: bool = True
) -> List[Tuple[int, int]]:
    """
    画像中心付近にスキャン位置を並べる格子状スキャンパターンを生成。

    Args:
        image_size: 画像サイズ（正方形）
        probe_size: プローブサイズ（正方形）
        step: スキャン間隔（ピクセル）
        num_points_y: Y方向のスキャン数
        num_points_x: X方向のスキャン数
        jitter: ジッターの最大幅（±jitter範囲でずらす）
        seed: 乱数シード
        bounds_check: プローブが画像からはみ出さないよう制限するか

    Returns:
        スキャン位置のリスト [(y, x), ...]
    """
    positions: List[Tuple[int, int]] = []
    rng = get_rng(seed)

    # スキャン領域の中心を画像中心に合わせる
    start_y = image_size // 2 - (num_points_y - 1) * step // 2
    start_x = image_size // 2 - (num_points_x - 1) * step // 2

    for i in range(num_points_y):
        for j in range(num_points_x):
            y_pos = start_y + i * step
            x_pos = start_x + j * step

            if jitter > 0:
                y_pos += randint(rng, -jitter, jitter + 1)
                x_pos += randint(rng, -jitter, jitter + 1)

            if bounds_check:
                if (x_pos - probe_size // 2 < 0 or x_pos + probe_size // 2 > image_size or
                    y_pos - probe_size // 2 < 0 or y_pos + probe_size // 2 > image_size):
                    continue

            positions.append((y_pos, x_pos))

    return positions