# utils/engines/difference_map.py
from ptychoep.backend.backend import np
from ptychoep.rng.rng_utils import get_rng, normal
from ptychoep.ptycho.projector import Fourier_projector

def _normalize_index_to_arrays(idx, xp):
    """
    Normalize indexing formats to flat array indices.

    Converts a 2D slice-based index (slice, slice) or tuple of arrays (yy, xx)
    into flattened (yy_flat, xx_flat) arrays for consistent scatter/gather operations.

    This is especially useful in algorithms like Difference Map where object updates
    rely on scatter-add operations over overlapping scan regions.

    Args:
        idx (tuple): A tuple of either (slice, slice) or (ndarray, ndarray) indexers.
        xp: The backend module (numpy or cupy).

    Returns:
        tuple: A pair of 1D arrays (yy, xx) representing flattened y- and x-coordinates.
    """
    y_idx, x_idx = idx
    if isinstance(y_idx, slice) and isinstance(x_idx, slice):
        y = xp.arange(y_idx.start, y_idx.stop)
        x = xp.arange(x_idx.start, x_idx.stop)
        YY, XX = xp.meshgrid(y, x, indexing="ij")
        return YY.reshape(-1), XX.reshape(-1)
    else:
        return y_idx.reshape(-1), x_idx.reshape(-1)

def _gather_patch(obj, idx, xp):
    """
    Safely extract a patch from the object array using flexible indexing.

    Supports both slice-based and advanced indexing (with arrays), and returns
    the sub-region of the object array corresponding to the scan position.

    Args:
        obj (ndarray): The full complex object array.
        idx (tuple): A tuple of (slice, slice) or (ndarray, ndarray) specifying the patch.
        xp: The backend module (numpy or cupy).

    Returns:
        ndarray: The selected sub-region of the object array.
    """

    y_idx, x_idx = idx
    if isinstance(y_idx, slice) and isinstance(x_idx, slice):
        return obj[y_idx, x_idx]
    else:
        return obj[y_idx, x_idx]
    

class DifferenceMap:
    """
    Difference Map algorithm for ptychographic phase retrieval.

    This implementation follows the standard formulation of the Difference Map (DM)
    approach, which iteratively refines estimates of the object and probe
    by alternating between real-space and Fourier-space constraints.

    The method maintains and updates the object and probe via scatter-add style
    averaging over overlapping patches in the object domain. Each iteration consists
    of applying a projection operator in the Fourier domain and combining it with
    previous estimates to enforce consistency with measured diffraction data.

    Attributes:
        ptycho (Ptycho): Ptycho object containing scan positions and diffraction data.
        beta (float): Relaxation parameter for the update.
        obj (ndarray): Current estimate of the object.
        prb (ndarray): Current estimate of the probe.
        callback (callable): Optional function to monitor convergence at each iteration.
        n_scan (int): Number of scan positions.
        prb_len (int): Size of the square probe region.
        diffs (ndarray): Stacked measured diffraction amplitudes.
        indices (List[slice or arrays]): Index mappings for scan regions.
        all_yy, all_xx (ndarray): Flattened scan region indices for scatter-add operations.

    Notes:
        - The computational cost is dominated by FFT and scatter operations.
        - The implementation supports both slice-based and array-based indexing of patches.
        - The update rules are based on the original Difference Map formulation used in ptychography.
    """

    def __init__(self, ptycho, beta=1.0, obj_init=None, prb_init=None, callback=None, seed : int = None):
        self.xp = np()
        self.ptycho = ptycho
        self.beta = beta
        self.callback = callback

        # --- init object/probe ---
        xp = self.xp
        if obj_init is None:
            rng = get_rng(seed)
            self.obj = normal(rng, mean=0.0, var=1.0,
                              size=(ptycho.obj_len, ptycho.obj_len),
                              dtype=xp.complex64)
        else:
            self.obj = xp.array(obj_init)
        self.prb = xp.array(prb_init) if prb_init is not None else xp.array(ptycho.prb.copy())

        # --- data ---
        self.diffs = xp.stack([d.diffraction for d in ptycho._diff_data])
        self.indices = [d.indices for d in ptycho._diff_data]
        self.prb_len = ptycho.prb_len
        self.n_scan = len(self.diffs)

        self.fft2 = xp.fft.fft2
        self.ifft2 = xp.fft.ifft2

        yyxx = [_normalize_index_to_arrays(idx, xp) for idx in self.indices]
        self.all_yy = xp.concatenate([yy for yy, _ in yyxx])
        self.all_xx = xp.concatenate([xx for _, xx in yyxx])

    def run(self, n_iter=100):
        xp = self.xp
        exit_waves = self._compute_exit_waves()
        Phi, _ = Fourier_projector(exit_waves, self.diffs)

        for it in range(n_iter):
            if self.callback:
                _, err = Fourier_projector(exit_waves, self.diffs, return_per_scan=False)
                self.callback(it, float(err), self.obj)

            self._update_object_probe(Phi)
            exit_waves = self._compute_exit_waves()
            Phi = Phi + Fourier_projector(2 * exit_waves - Phi, self.diffs)[0] - exit_waves

        return self.obj, self.prb

    def _compute_exit_waves(self):
        xp = self.xp
        patches = xp.stack([_gather_patch(self.obj, idx, xp) for idx in self.indices])
        return self.prb[None, :, :] * patches

    def _update_object_probe(self, Phi):
        xp = self.xp

        # --- object update (scatter add) ---
        num_obj_real = xp.zeros_like(self.obj.real)
        num_obj_imag = xp.zeros_like(self.obj.imag)
        den_obj = xp.zeros_like(self.obj.real) + 1e-10

        prb_conj = self.prb.conj()
        prb_abs2 = xp.abs(self.prb) ** 2

        all_val = xp.concatenate([(prb_conj * phi).reshape(-1) for phi in Phi])
        all_prb_abs2 = xp.concatenate([prb_abs2.reshape(-1) for _ in Phi])

        xp.add.at(num_obj_real, (self.all_yy, self.all_xx), all_val.real)
        xp.add.at(num_obj_imag, (self.all_yy, self.all_xx), all_val.imag)
        xp.add.at(den_obj, (self.all_yy, self.all_xx), all_prb_abs2)

        self.obj = (num_obj_real + 1j * num_obj_imag) / den_obj

        # --- probe update ---
        num_prb = xp.zeros_like(self.prb)
        den_prb = xp.zeros_like(self.prb) + 1e-10
        for idx, phi in zip(self.indices, Phi):
            obj_patch = _gather_patch(self.obj, idx, xp)
            num_prb += obj_patch.conj() * phi
            den_prb += xp.abs(obj_patch) ** 2
        self.prb = num_prb / den_prb
