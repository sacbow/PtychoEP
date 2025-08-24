from ptychoep.backend.backend import np
from ptychoep.rng.rng_utils import get_rng, normal
from ptychoep.ptycho.core import Ptycho
from ptychoep.ptycho.projector import Fourier_projector

class BasePIE:
    """
    Abstract base class for PIE-type ptychographic reconstruction algorithms.

    This class provides a common framework for phase retrieval methods based on the 
    Ptychographical Iterative Engine (PIE), such as ePIE and rPIE. It defines the 
    overall optimization loop and interface for updating the object function, while 
    allowing specific update rules to be implemented in derived classes.

    Attributes:
        ptycho (Ptycho): Container for scan positions, probe, and diffraction data.
        alpha (float): Step size parameter for object update.
        obj (ndarray): Complex-valued object array (reconstruction target).
        prb (ndarray): Complex-valued probe array (copied from input ptycho).
        fft2, ifft2: Fourier transform functions based on the current backend (numpy or cupy).
        callback (callable): Optional function called after each iteration: callback(it, err, obj).

    Args:
        ptycho (Ptycho): Ptycho object with probe, object size, and scan data configured.
        alpha (float): Object update step size.
        obj_init (ndarray or None): Optional initial guess for the object. If None, initialized randomly.
        dtype: Data type for internal arrays (default: complex64).
        callback (callable or None): Optional callback function for logging or visualization.
        seed (int or None): Random seed for reproducible initialization (if obj_init is None).

    Methods:
        run(n_iter=100): Executes the reconstruction for a given number of iterations.
        _update_object(...): Abstract method to be implemented in subclasses to define
                             how the object is updated at each scan position.
    """

    def __init__(self, ptycho: Ptycho, alpha: float = 0.1, obj_init=None, dtype = np().complex64, callback=None, seed : int = None):
        self.xp = np() 
        self.ptycho = ptycho
        self.alpha = self.xp.asarray(alpha)
        self.callback = callback
        self.dtype = dtype

        # Initializing object
        if obj_init is None:
            rng = get_rng(seed)
            self.obj = normal(rng, mean=0.0, var=1.0, size=(ptycho.obj_len, ptycho.obj_len), dtype=self.dtype)
        else:
            self.obj = self.xp.array(obj_init)

        # Set probe
        self.prb = self.xp.array(ptycho.prb.copy())
        # FFT
        self.fft2 = self.xp.fft.fft2
        self.ifft2 = self.xp.fft.ifft2

    def run(self, n_iter=100):
        for it in range(n_iter):
            err = 0.0
            for d in self.ptycho._diff_data:
                yy, xx = d.indices
                obj_patch = self.obj[yy, xx]
                exit_wave = self.prb * obj_patch

                proj_wave, error_val = Fourier_projector(exit_wave, d.diffraction)
                err += error_val

                self._update_object(proj_wave, exit_wave, (yy, xx))

            avg_err = float(err / len(self.ptycho._diff_data))

            if self.callback:
                self.callback(it, avg_err, self.obj)

        return self.obj


    def _update_object(self, proj_wave, exit_wave, indices):
        raise NotImplementedError("派生クラスで実装してください")
