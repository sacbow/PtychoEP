import pytest
from PtychoEP.backend.backend import set_backend, np as backend_np
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.ptycho.aperture_utils import circular_aperture
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.ptychoep.core import PtychoEP

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_ptycho_ep_end_to_end(backend):
    set_backend(backend)
    xp = backend_np()

    # --- Load synthetic object and probe ---
    obj_real = load_data_image("lily.png")
    obj_phase = load_data_image("moon.png")
    obj = obj_real * xp.exp(1j * (xp.pi / 2) * obj_phase)

    probe = circular_aperture(size=64, r=0.5)

    # --- Build Ptycho object ---
    ptycho = Ptycho()
    ptycho.set_object(obj)
    ptycho.set_probe(probe)

    # --- Generate scan positions and simulate measurements ---
    positions = generate_spiral_scan_positions(
        image_size=512,
        probe_size=64,
        step=24.0,
        num_points=100
    )
    ptycho.forward_and_set_diffraction(positions)
    ptycho.sort_diffraction_data(key="center_distance")

    # --- Run EP with 1 iteration ---
    errors = []
    def callback(i, err, est):
        errors.append(err)

    ep_solver = PtychoEP(
        ptycho=ptycho,
        damping=0.8,
        callback=callback
    )

    est_obj, est_obj_precision = ep_solver.run(n_iter=1)

    # --- Basic output checks ---
    assert est_obj.shape == obj.shape
    assert xp.iscomplexobj(est_obj)
    assert xp.isrealobj(est_obj_precision)
    assert len(errors) == 1
