import pytest
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray as UA, fft_ua, ifft_ua
from PtychoEP.utils.engines.ptycho_ep.fft_channel import FFTChannel
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.rng_utils import get_rng

@pytest.fixture(autouse=True, params=["numpy", "cupy"])
def setup_backend(request):
    set_backend(request.param)

def test_fft_channel_forward_and_backward_consistency():
    rng_init = get_rng()
    shape = (16, 16)
    rng = get_rng()
    input_msg = UA.normal(shape=shape, rng=rng, scalar_precision = False)

    channel = FFTChannel(rng = rng_init)
    channel.receive_msg_from_input(input_msg)

    # 出力メッセージをランダムに用意してbackward側に与える
    dummy_output_msg = UA.normal(shape=shape, rng=rng, scalar_precision = True)
    channel.receive_msg_from_output(dummy_output_msg)

    # backwardでObjectへのメッセージを生成
    back_msg = channel.backward()
    assert isinstance(back_msg, UA)
    assert back_msg.mean.shape == shape
    assert back_msg.precision.shape == () or back_msg.scalar_precision

    # forwardでもう一度伝搬（belief/output_msgがセットされている）
    fwd_msg = channel.forward()
    assert isinstance(fwd_msg, UA)
    assert fwd_msg.mean.shape == shape

def test_forward_before_backward_returns_random():
    rng_init = get_rng()
    shape = (8, 8)
    rng = get_rng()
    input_msg = UA.normal(shape=shape, rng=rng)

    channel = FFTChannel(rng_init)
    channel.receive_msg_from_input(input_msg)

    # backwardが行われていない状態では、forwardはランダムなUAを返す
    out = channel.forward()
    assert isinstance(out, UA)
    assert out.mean.shape == shape

def test_backward_without_inputs_raises():
    rng_init = get_rng()
    channel = FFTChannel(rng_init)
    with pytest.raises(RuntimeError):
        channel.backward()
