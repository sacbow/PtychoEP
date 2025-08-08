# test/test_probe.py
import pytest
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray as UA
from PtychoEP.utils.engines.ptycho_ep.probe import Probe
from PtychoEP.utils.rng_utils import get_rng

@pytest.fixture(autouse=True, params=["numpy", "cupy"])
def setup_backend(request):
    set_backend(request.param)

def test_forward_backward_consistency():
    xp = backend_np()
    rng = get_rng()
    shape = (8, 8)

    # ランダムなProbeデータ（ゼロ除算回避のため非ゼロ）
    probe_data = xp.random.rand(*shape) + 1j * xp.random.rand(*shape)
    probe_data = probe_data.astype(xp.complex64) + 0.1

    probe = Probe(probe_data)

    # ランダムUA
    ua_in = UA.normal(shape=shape, rng=rng)

    # forward → backward がほぼ ua_in に戻るか
    ua_fwd = probe.forward(ua_in)
    ua_back = probe.backward(ua_fwd)

    assert xp.allclose(ua_back.mean, ua_in.mean, atol=1e-6)
    assert xp.allclose(ua_back.precision, ua_in.precision, atol=1e-6)

def test_set_data_and_shape_check():
    xp = backend_np()
    probe = Probe()

    # 正しい2DデータはOK
    data = xp.ones((4, 4), dtype=xp.complex64)
    probe.set_data(data)
    assert probe.data.shape == (4, 4)

    # 1Dデータはエラー
    with pytest.raises(ValueError):
        probe.set_data(xp.ones((4,), dtype=xp.complex64))

def test_shape_mismatch_raises():
    xp = backend_np()
    probe = Probe(xp.ones((4, 4), dtype=xp.complex64))
    ua_wrong_shape = UA.normal(shape=(3, 3), rng=get_rng())

    with pytest.raises(ValueError):
        probe.forward(ua_wrong_shape)

    with pytest.raises(ValueError):
        probe.backward(ua_wrong_shape)
