import pytest
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.engines.ptycho_ep.object import Object
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray
from PtychoEP.utils.engines.ptycho_ep.accumulative_uncertain_array import AccumulativeUncertainArray
from PtychoEP.utils.ptycho.data import DiffractionData


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_object_registration_and_belief(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (8, 8)
    patch = (slice(2, 6), slice(2, 6))

    obj_init = xp.ones(shape, dtype=xp.complex64)
    prb_init = xp.ones((4, 4), dtype=xp.complex64)

    obj = Object(shape=shape, rng=None, initial_probe=prb_init, initial_object=obj_init)

    dummy_data = DiffractionData(diffraction=xp.ones((4, 4)), position=(4, 4))
    dummy_data.indices = patch

    obj.register_data(dummy_data)

    assert dummy_data in obj.data_registry
    assert dummy_data in obj.probe_registry
    assert dummy_data in obj.msg_from_data

    belief = obj.get_belief()
    assert isinstance(belief, UncertainArray)
    assert belief.mean.shape == shape
    assert not belief.scalar_precision


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_object_forward_backward_cycle(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (8, 8)
    patch = (slice(2, 6), slice(2, 6))

    obj = Object(shape=shape, rng=None, initial_probe=xp.ones((4, 4)), initial_object=xp.ones(shape, dtype=xp.complex64))
    data = DiffractionData(diffraction=xp.ones((4, 4)), position=(4, 4))
    data.indices = patch
    obj.register_data(data)

    # Forward should update probe.input_belief
    obj.forward(data)
    prb = obj.probe_registry[data]
    assert isinstance(prb.input_belief, UncertainArray)

    # Manually simulate backward msg from probe
    prb.msg_to_object = prb.input_belief.scaled(2.0)
    old_msg = obj.msg_from_data[data].mean.copy()

    # Backward updates belief and message
    obj.backward(data)
    new_msg = obj.msg_from_data[data]
    assert xp.allclose(new_msg.mean, old_msg)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_object_prior_message_handling(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (6, 6)
    obj = Object(shape=shape, rng=None, initial_probe=xp.ones((4, 4)))

    prior_msg = UncertainArray(xp.ones(shape, dtype=xp.complex64), 2.0).to_array_precision()
    obj.receive_msg_from_prior(prior_msg)

    # Check belief has been updated
    belief = obj.get_belief()
    assert isinstance(belief, UncertainArray)
    assert not belief.scalar_precision
    assert xp.allclose(obj.msg_from_prior.mean, 1.0)

    # send_msg_to_prior returns division of belief by prior
    msg = obj.send_msg_to_prior()
    assert isinstance(msg, UncertainArray)
    assert msg.shape == shape
