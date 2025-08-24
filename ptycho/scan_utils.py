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
    Generate a Fermat spiral scan pattern for ptychography.

    This function generates a list of scanning positions following a Fermat spiral trajectory,
    which offers more uniform coverage and higher overlap ratio compared to traditional grid or concentric scans.

    Args:
        image_size: Size of the square image in pixels.
        probe_size: Size of the square probe in pixels.
        num_points: Number of scan points to generate.
        step: Radial expansion factor of the spiral.
        center: Center of the scan. If None, the center of the image is used.
        bounds_check: If True, only positions fully inside the image boundary are included.

    Returns:
        List of (y, x) scan positions as tuples of integers.

    References:
        X. Huang et al., "Effects of overlap uniformness for ptychography," Opt. Express 22(11), 12634–12644 (2014).
        https://doi.org/10.1364/OE.22.012634
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
    Deprecated. Use generate_centered_grid_positions instead.
    """
    positions: List[Tuple[int, int]] = []
    rng = get_rng(seed)

    for y in range(0, image_size, step):
        for x in range(0, image_size, step):
            y_pos = y
            x_pos = x
            if jitter > 0:
                y_pos += randint(rng, -jitter, jitter + 1)
                x_pos += randint(rng, -jitter, jitter + 1)
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
    Generate a regular grid of scan positions centered in the image.

    Args:
        image_size (int): Size of the square object image.
        probe_size (int): Size of the square probe.
        step (int): Distance between adjacent scan points (in pixels).
        num_points_y (int): Number of scan points along Y-axis.
        num_points_x (int): Number of scan points along X-axis.
        jitter (int): Maximum random displacement added to each scan position.
        seed (int, optional): Random seed for jitter reproducibility.
        bounds_check (bool): Whether to exclude positions where probe exceeds image boundary.

    Returns:
        List[Tuple[int, int]]: List of (y, x) scan positions.
    """
    positions: List[Tuple[int, int]] = []
    rng = get_rng(seed)

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
