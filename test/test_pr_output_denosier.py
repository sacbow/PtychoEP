import pytest
import numpy as raw_np
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.ptycho.data import DiffractionData
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray as UA
from PtychoEP.utils.engines.ptycho_ep.denoiser import PROutputDenoiser


@pytest.fixture(autouse=True, params=["numpy", "cupy"])
def setup_backend(request):
    """NumPy / CuPy 切り替え"""
    set_backend(request.param)


def make_dummy_data(shape=(16, 16)):
    """簡単なDiffractionDataを作成"""
    amp = backend_np().ones(shape, dtype=backend_np().float32)
    phase = backend_np().zeros(shape, dtype=backend_np().float32)
    diffraction = amp * backend_np().exp(1j * phase)

    return DiffractionData(
        position=(0, 0),
        diffraction=diffraction,
        meta={},
        indices=None
    )


def test_forward_msg_basic():
    shape = (16, 16)
    data = make_dummy_data(shape)
    gamma_w = 1.0
    denoiser = PROutputDenoiser(shape=shape, data=data, gamma_w=gamma_w)

    # 入力メッセージ（mean = 1 + 0j, precision = 1.0）
    mean = backend_np().ones(shape, dtype=backend_np().complex64)
    precision = backend_np().ones(shape, dtype=backend_np().float32)
    msg = UA(mean=mean, precision=precision)

    denoiser.receive_msg(msg)
    out_msg = denoiser.forward_msg()

    # 型チェック
    assert isinstance(out_msg, UA)
    assert out_msg.mean.shape == shape
    assert out_msg.scalar_precision or out_msg.precision.shape == shape

    # 精度がすべて正
    assert backend_np().all(out_msg.precision > 0)

    # MSE誤差の確認
    assert isinstance(denoiser.error, float)
    assert denoiser.error >= 0