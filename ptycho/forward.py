from typing import List, Tuple
from ptychoep.backend.backend import np
from ptychoep.ptycho.data import DiffractionData
from .core import Ptycho


def generate_diffraction(ptycho: Ptycho, positions: List[Tuple[int, int]]) -> List[DiffractionData]:
    """
    Generate noiseless diffraction data based on the forward model of ptychography.

    This function extracts patches from the object at specified scan positions,
    multiplies them by the probe, applies 2D FFT, and stores the resulting 
    diffraction amplitude (no phase) in a DiffractionData container.

    Parameters
    ----------
    ptycho : Ptycho
        A Ptycho object with object and probe properly initialized.
    positions : List[Tuple[int, int]]
        List of scan coordinates, where each entry is a tuple (y, x).

    Returns
    -------
    List[DiffractionData]
        A list of DiffractionData instances, each corresponding to a scan position
        and holding the generated diffraction image (noiseless).
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

        # Extract object patch corresponding to the scan position
        obj_patch = ptycho.obj[
            y - prb_len // 2 : y + prb_len // 2,
            x - prb_len // 2 : x + prb_len // 2
        ]

        # Compute diffraction amplitude via 2D FFT
        diff = np().abs(fft2(obj_patch * ptycho.prb, norm="ortho"))

        # Store slice indices for possible use in backward operations
        yy0, yy1 = y - prb_len // 2, y + prb_len // 2
        xx0, xx1 = x - prb_len // 2, x + prb_len // 2
        indices = (slice(yy0, yy1), slice(xx0, xx1))

        diffs.append(DiffractionData(position=pos, diffraction=diff, indices=indices))

    return diffs
