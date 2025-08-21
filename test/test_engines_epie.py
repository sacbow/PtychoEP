import pytest
from PtychoEP.backend.backend import set_backend, np
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.classic_engines.epie import ePIE

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_epie_runs_and_reduces_error(backend):
    set_backend(backend)
    ptycho = Ptycho()
    obj = np().array(load_data_image("cameraman.png")) * np().exp(1j * np().pi * np().array(load_data_image("eagle.png")), dtype = np().complex64)
    probe = np().array(load_data_image("probe.png"), dtype = np().complex64)
    ptycho.set_object(obj)
    ptycho.set_probe(probe)
    positions = generate_spiral_scan_positions(image_size=512, probe_size=128, num_points=50)
    ptycho.forward_and_set_diffraction(positions)

    errors = []
    def callback(iter_idx, err, obj_est):
        errors.append(err)

    epie = ePIE(ptycho, alpha=0.1, beta=0.1, callback=callback)
    epie.run(n_iter=10)

    assert len(errors) == 10
    assert errors[0] > errors[-1]
