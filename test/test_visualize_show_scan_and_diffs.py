import pytest
import matplotlib
matplotlib.use("Agg")  # GUI不要のバックエンド
import matplotlib.pyplot as plt

from ptychoep.backend.backend import set_backend, np as backend_np
from ptychoep.ptycho.core import Ptycho
from ptychoep.ptycho.forward import generate_diffraction
from ptychoep.ptycho.visualize import show_scan_and_diffs

@pytest.fixture(autouse=True)
def setup_backend():
    set_backend("numpy")

def create_simple_ptycho_with_diffs():
    p = Ptycho()
    obj = backend_np().ones((64, 64), dtype=backend_np().complex64)
    prb = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(prb)
    positions = [(32, 32), (40, 40), (48, 48)]
    diffs = generate_diffraction(p, positions)
    p.set_diffraction_from_forward(diffs)
    return p

def test_show_scan_and_diffs_runs(monkeypatch):
    p = create_simple_ptycho_with_diffs()
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = show_scan_and_diffs(p, num_patterns=2, log_scale=True)
    assert fig is not None
    assert len(fig.axes) == 3
