from ptychoep.backend.backend import set_backend, np as backend_np
from ptychoep.ptychoep.uncertain_array import UncertainArray
from ptychoep.ptychoep.object import Object
from ptychoep.ptychoep.probe import Probe
from ptychoep.ptycho.data import DiffractionData
import pytest

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_probe_init_and_set_data(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (4, 4)

    # Dummy diffraction data
    dummy_diff = DiffractionData(diffraction=xp.ones(shape), position=(0, 0))
    dummy_diff.indices = (slice(0, 4), slice(0, 4))

    # Dummy object (parent) with object_init
    obj = Object(shape=(4, 4), rng=None, initial_probe=xp.ones(shape, dtype=xp.complex64),
                 initial_object=xp.ones((4, 4), dtype=xp.complex64))

    prb = Probe(data=xp.ones(shape, dtype=xp.complex64), parent=obj, diffraction=dummy_diff)
    assert prb.data.shape == shape
    assert prb.abs2.shape == shape
    assert xp.all(prb.abs2 > 0)
    assert prb.data_inv.shape == shape
    assert prb.child is not None
    assert prb.child.probe is prb

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_probe_forward_and_backward(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (4, 4)

    # Dummy diffraction data
    dummy_diff = DiffractionData(diffraction=xp.ones(shape), position=(0, 0))
    dummy_diff.indices = (slice(0, 4), slice(0, 4))

    # Dummy object (parent)
    obj = Object(shape=(4, 4), rng=None, initial_probe=xp.ones(shape, dtype=xp.complex64),
                 initial_object=xp.ones((4, 4), dtype=xp.complex64))

    probe = Probe(data=xp.ones(shape, dtype=xp.complex64), parent=obj, diffraction=dummy_diff)

    input_mean = xp.ones(shape, dtype=xp.complex64)
    input_prec = xp.ones(shape, dtype=xp.float32)
    probe.input_belief = UncertainArray(mean=input_mean, precision=input_prec)

    # Forward
    probe.forward()
    fft_in = probe.child.input_belief
    assert xp.allclose(fft_in.mean, input_mean * probe.data)
    expected_prec = input_prec / xp.abs(probe.data) ** 2
    assert xp.allclose(fft_in.precision, expected_prec.astype(xp.float32))

    # Simulate backward
    probe.child.msg_to_probe = fft_in
    probe.backward()
    back_msg = probe.msg_to_object
    assert xp.allclose(back_msg.mean, fft_in.mean * probe.data_inv)
    expected_back_prec = fft_in.precision / xp.abs(probe.data_inv) ** 2
    assert xp.allclose(back_msg.precision, expected_back_prec.astype(xp.float32))
