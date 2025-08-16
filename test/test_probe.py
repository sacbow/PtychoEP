import pytest
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray
from PtychoEP.utils.engines.ptycho_ep.probe import Probe
from PtychoEP.utils.ptycho.data import DiffractionData


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_probe_init_and_set_data(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (4, 4)
    data = xp.ones(shape, dtype=xp.complex64)

    prb = Probe(data)
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

    probe_data = xp.ones((4, 4), dtype=xp.complex64)
    input_mean = xp.ones((4, 4), dtype=xp.complex64)
    input_prec = xp.ones((4, 4), dtype=xp.float32)

    # Create probe
    probe = Probe(data=probe_data)
    probe.input_belief = UncertainArray(mean=input_mean, precision=input_prec)

    # Simulate forward pass
    probe.forward()
    fft_in = probe.child.input_belief

    # Check forward scaled result
    assert xp.allclose(fft_in.mean, input_mean * probe_data)
    expected_prec_forward = input_prec / xp.abs(probe_data)**2
    assert xp.allclose(fft_in.precision, expected_prec_forward.astype(xp.float32))

    # Simulate FFT output -> msg_to_probe
    probe.child.msg_to_probe = fft_in  # normally set by FFTChannel.backward()
    probe.backward()

    # Check backward scaled result
    back_msg = probe.msg_to_object
    assert xp.allclose(back_msg.mean, fft_in.mean * probe.data_inv)
    expected_prec_back = fft_in.precision / xp.abs(probe.data_inv)**2
    assert xp.allclose(back_msg.precision, expected_prec_back.astype(xp.float32))