import pytest
from PtychoEP.backend.backend import set_backend, np as backend_np
from PtychoEP.utils.io_utils import load_data_image
from PtychoEP.ptycho.aperture_utils import circular_aperture
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.ptychoep.object import Object
from PtychoEP.ptychoep.probe_updater import ProbeUpdater

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_probe_updater_runs_and_updates_probe(backend):
    set_backend(backend)
    xp = backend_np()

    # --- Load object and probe ---
    obj_real = load_data_image("lily.png")
    obj_phase = load_data_image("moon.png")
    obj = obj_real * xp.exp(1j * (xp.pi / 2) * obj_phase)
    probe = circular_aperture(size=64, r=0.5)

    # --- Build Ptycho object and simulate diffraction ---
    ptycho = Ptycho()
    ptycho.set_object(obj)
    ptycho.set_probe(probe)
    positions = generate_spiral_scan_positions(
        image_size=512,
        probe_size=64,
        step=24.0,
        num_points=20,  # 少し軽め
    )
    ptycho.forward_and_set_diffraction(positions)

    # --- Build Object node and ProbeUpdater ---
    obj_node = Object(
        shape=obj.shape,
        rng=None,
        initial_probe=probe,
        initial_object=obj
    )
    for diff in ptycho._diff_data:
        obj_node.register_data(diff)

    updater = ProbeUpdater(obj_node)

    # --- Record original probe data ---
    old_data = [p.data.copy() for p in obj_node.probe_registry.values()]

    # --- Run one forward pass to initialize msg_to_probe ---
    for diff in ptycho._diff_data:
        obj_node.forward(diff)  # Object → Probe
        probe = obj_node.probe_registry[diff]
        probe.forward()
        probe.child.forward()
        probe.child.likelihood.backward()
        probe.child.backward()
        probe.backward()
        obj_node.backward(diff)
        
    # --- Run probe update ---
    updater.update(n_iter=1)

    # --- Check that probe data changed ---
    for old, probe in zip(old_data, obj_node.probe_registry.values()):
        new = probe.data
        assert xp.any(xp.abs(new - old) > 1e-6), "Probe data did not change"

    # --- Check that precision was updated ---
    for probe in obj_node.probe_registry.values():
        precision = probe.child.msg_from_likelihood.precision
        assert precision > 0
