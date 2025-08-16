import pytest
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray
from PtychoEP.utils.engines.ptycho_ep.denoiser import Denoiser
from PtychoEP.utils.engines.ptycho_ep.fft_channel import FFTChannel
from PtychoEP.utils.ptycho.data import DiffractionData


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_denoiser_compute_belief_and_backward(backend):
    set_backend(backend)
    xp = backend_np()
    
    # --- Setup: fake data and FFTChannel ---
    shape = (4, 4)
    amplitude = xp.ones(shape, dtype=xp.float32) * 2.0   # y = 2.0 (observed amp)
    gamma_w = 1.0

    diff = DiffractionData(diffraction=amplitude, position=(0, 0))  # squared for intensity
    diff.indices = (slice(0, 4), slice(0, 4))
    diff.gamma_w = gamma_w
    fake_probe = type("FakeProbe", (), {"parent": type("FakeObject", (), {"object_init": xp.ones(shape)})(), "data": xp.ones(shape), "diff": diff})()
    fft_channel = FFTChannel(parent_probe=fake_probe, diff=diff)

    # --- Override fft_channel.msg_from_denoiser to avoid fft init ---
    initial_z0 = xp.ones(shape, dtype=xp.complex64)  # |z0| = 1
    msg_fft = UncertainArray(mean=initial_z0, precision=2.0)
    fft_channel.msg_from_denoiser = msg_fft.copy()

    # --- Connect Denoiser ---
    denoiser = Denoiser(diff=diff, parent=fft_channel)
    denoiser.msg_from_fft = msg_fft

    # --- Run compute_belief ---
    denoiser.compute_belief()
    belief = denoiser.belief

    assert belief.mean.shape == shape
    assert belief.precision.shape == shape
    assert not belief.scalar_precision
    assert xp.all(xp.abs(belief.mean) > 0)

    # --- Error (mean square amp difference) ---
    assert denoiser.error >= 0.0

    # --- Run backward and check damping behavior ---
    fft_channel.msg_from_denoiser = msg_fft.copy()  # previous msg for damping
    denoiser.backward()
    updated_msg = fft_channel.msg_from_denoiser

    assert isinstance(updated_msg, UncertainArray)
    assert xp.allclose(msg_fft.mean, updated_msg.mean, rtol=0.5)
    assert xp.all(updated_msg.precision > 0)
