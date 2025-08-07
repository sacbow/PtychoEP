import pytest
from PtychoEP.utils.engines.ptycho_ep.object import Object
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray as UA
from PtychoEP.utils.ptycho.data import DiffractionData
from PtychoEP.utils.backend import set_backend, np
from PtychoEP.utils.rng_utils import get_rng


@pytest.fixture(autouse=True)
def setup_backend():
    set_backend("numpy")

@pytest.fixture
def dummy_data():
    shape = (32, 32)
    indices = (slice(10, 26), slice(10, 26))
    diffraction = np().ones((16, 16), dtype=np().complex64)
    return DiffractionData(position=(0, 0), diffraction=diffraction, indices=indices)

def test_register_and_receive(dummy_data):
    obj = Object(shape=(32, 32), rng=get_rng())
    obj.register_data(dummy_data)
    
    assert dummy_data in obj.msg_from_data
    assert dummy_data in obj.data_registry

    # make dummy message and apply
    msg = UA.normal(shape=(16, 16), rng=get_rng())
    obj.receive_msg_from_data(dummy_data, msg)
    patch = obj.get_patch_ua(dummy_data)

    # patch should be close to msg (since belief = only msg)
    assert np().allclose(patch.mean, msg.mean/2., atol=1e-5)
    assert np().allclose(patch.precision, msg.precision + 1, atol=1e-5)

def test_send_msg_to_prior(dummy_data):
    obj = Object(shape=(32, 32), rng=get_rng())
    obj.register_data(dummy_data)
    obj.receive_msg_from_prior(UA.zeros((32, 32), scalar_precision = False))

    output_msg = obj.send_msg_to_prior()
    belief = obj.get_belief()

    # send_msg_to_prior = belief / prior → should recover belief if prior=1
    assert np().allclose(output_msg.mean[0:5,0:5], belief.mean[0:5,0:5])
