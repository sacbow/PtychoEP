import pytest
from ptychoep.ptycho.core import Ptycho
from ptychoep.ptycho.scan_utils import generate_grid_scan_positions
from ptychoep.ptycho.aperture_utils import circular_aperture
from ptychoep.utils.io_utils import load_data_image
from ptychoep.ptychoep.core import PtychoEP
from ptychoep.ptychoep.prior import SparsePrior
from ptychoep.backend.backend import np


def make_dummy_ptycho(image_size=128, probe_size=64, step=32, sparsity=0.2):
    """
    Create a dummy Ptycho object with synthetic object and probe,
    and register diffraction data using grid scan positions.
    """
    obj = load_data_image("lily.png", normalize=True)
    obj = obj[:image_size, :image_size]  # crop if necessary
    obj = obj + 1j * load_data_image("moon.png", normalize=True)[:image_size, :image_size]

    prb = circular_aperture(size=probe_size, r=0.5)

    ptycho = Ptycho()
    ptycho.set_object(obj)
    ptycho.set_probe(prb)

    positions = generate_grid_scan_positions(
        image_size=image_size,
        probe_size=probe_size,
        step=step,
        jitter=0,
        bounds_check=True
    )
    ptycho.forward_and_set_diffraction(positions)
    return ptycho



def test_sparse_prior_forward_runs():
    ptycho = make_dummy_ptycho()
    engine = PtychoEP(ptycho, prior_name="sparse", sparsity=0.2)
    prior = engine.obj_node.prior

    assert isinstance(prior, SparsePrior)
    prior.forward()
    assert prior.belief is not None
    assert prior.msg_to_object is not None


def test_sparse_prior_belief_properties():
    ptycho = make_dummy_ptycho()
    engine = PtychoEP(ptycho, prior_name="sparse", sparsity=0.2)
    prior = engine.obj_node.prior
    prior.forward()

    belief = prior.belief
    assert (belief.precision > 0).all()
    assert not np().any(np().isnan(belief.mean))


def test_sparse_prior_effect_strength():
    ptycho = make_dummy_ptycho()

    ep1 = PtychoEP(ptycho, prior_name="sparse", sparsity=0.05)
    ep1.obj_node.prior.forward()
    var1 = 1.0 / ep1.obj_node.prior.belief.precision

    ep2 = PtychoEP(ptycho, prior_name="sparse", sparsity=0.5)
    ep2.obj_node.prior.forward()
    var2 = 1.0 / ep2.obj_node.prior.belief.precision

    # variance should on average be different depending on sparsity level
    diff = np().mean(np().abs(var1 - var2))
    assert diff > 1e-4


def test_no_prior_means_prior_is_none():
    ptycho = make_dummy_ptycho()
    engine = PtychoEP(ptycho, prior_name="gaussian")  # special case
    assert engine.obj_node.prior is None
