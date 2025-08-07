import pytest
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray as UA
from PtychoEP.utils.engines.ptycho_ep.denoiser import GaussianPriorDenoiser, SparsePriorDenoiser
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.rng_utils import get_rng

@pytest.fixture(autouse=True)
def setup_backend():
    """テストはnumpyバックエンドで固定"""
    set_backend("numpy")

@pytest.fixture
def sample_input_msg():
    shape = (32, 32)
    rng = get_rng(seed=42)
    return UA.normal(shape=shape, rng=rng)

def test_gaussian_prior_forward_msg(sample_input_msg):
    shape = sample_input_msg.mean.shape
    denoiser = GaussianPriorDenoiser(shape=shape)
    denoiser.receive_msg(sample_input_msg)

    # forward message should be zeros
    fwd_msg = denoiser.forward_msg()
    assert backend_np().allclose(fwd_msg.mean, 0)
    assert backend_np().allclose(fwd_msg.precision, 1)

    # belief should be equal to input message (since prior = 0 mean, unit variance)
    denoiser.compute_belief()
    belief = denoiser.belief
    expected_mean = sample_input_msg.mean * sample_input_msg.precision / (1 + sample_input_msg.precision)
    expected_precision = sample_input_msg.precision + 1
    assert backend_np().allclose(belief.mean, expected_mean)
    assert backend_np().allclose(belief.precision, expected_precision)

def test_sparse_prior_forward_msg(sample_input_msg):
    shape = sample_input_msg.mean.shape
    rho = 0.2
    denoiser = SparsePriorDenoiser(shape=shape, rho=rho)
    denoiser.receive_msg(sample_input_msg)

    fwd_msg = denoiser.forward_msg()
    assert isinstance(fwd_msg, UA)

    # belief must be defined
    assert denoiser.belief is not None

    # precision must be strictly positive
    assert backend_np().all(denoiser.belief.precision > 0)

    # 0 <= pi <= 1 なので平均値が縮小される方向に働く（疎性）
    assert backend_np().all(backend_np().abs(denoiser.belief.mean) <= backend_np().abs(sample_input_msg.mean) + 1e-4)

