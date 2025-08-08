# test/test_ptycho_ep.py
import numpy as np
import pytest

from PtychoEP.utils.engines.ptycho_ep.core import PtychoEP
from PtychoEP.utils.rng_utils import get_rng
from PtychoEP.utils.ptycho.core import Ptycho
from PtychoEP.utils.ptycho.forward import generate_diffraction
from PtychoEP.utils.ptycho.noise import GaussianNoise
from PtychoEP.utils.ptycho.scan_utils import generate_spiral_scan_positions


def create_simple_ptycho():
    """64x64 の単純な物体と16x16プローブを持つ Ptycho を作成"""
    p = Ptycho()
    xp = np
    obj = xp.ones((64, 64), dtype=np.complex64)
    prb = xp.ones((16, 16), dtype=np.complex64)
    p.set_object(obj)
    p.set_probe(prb)

    positions = generate_spiral_scan_positions(image_size=64, probe_size=16, num_points=5)
    diffs = generate_diffraction(p, positions)
    # ノイズ付与（gamma_w セット）
    GaussianNoise(var=1e-4) @ p
    p.set_diffraction_from_forward(diffs)
    return p


def test_gaussian_prior_runs_and_returns_shapes():
    ptycho = create_simple_ptycho()

    solver = PtychoEP(ptycho, prior_type="gaussian", damping=0.9)
    assert solver.prior is None, "Gaussian priorの場合、priorはNoneであるべき"

    obj_est, prb_est = solver.run(n_iter=3)
    assert obj_est.shape == (ptycho.obj_len, ptycho.obj_len)
    assert prb_est.shape == (ptycho.prb_len, ptycho.prb_len)


def test_sparse_prior_runs_and_callback_called():
    ptycho = create_simple_ptycho()

    called = []
    def cb(it, err, est):
        called.append((it, err))

    solver = PtychoEP(
        ptycho, prior_type="sparse",
        prior_kwargs={"rho": 0.1},
        damping=0.9,
        callback=cb
    )
    obj_est, prb_est = solver.run(n_iter=3)

    assert solver.prior is not None
    assert len(called) == 3
    assert obj_est.shape == (ptycho.obj_len, ptycho.obj_len)
    assert prb_est.shape == (ptycho.prb_len, ptycho.prb_len)


def test_error_reduces_for_simple_case():
    ptycho = create_simple_ptycho()
    rng = get_rng(15)

    errors = []
    def cb(it, err, est):
        errors.append(err)

    solver = PtychoEP(
        ptycho, prior_type="gaussian",
        damping=0.9,
        callback=cb
    )
    solver.run(n_iter=20)

    # 最後の誤差が初期より小さいこと（単調減少は保証しない）
    assert errors[-1] <= errors[0]
