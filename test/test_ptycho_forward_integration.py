import pytest
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.ptycho.forward import generate_diffraction
from PtychoEP.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.backend.backend import set_backend, np as backend_np

@pytest.fixture(autouse=True)
def setup_backend():
    """テストはnumpyバックエンドで固定"""
    set_backend("numpy")

def create_simple_ptycho():
    """64x64の単純なオブジェクト・16x16のプローブを持つPtychoを生成"""
    p = Ptycho()
    obj = backend_np().ones((64, 64), dtype=backend_np().complex64)
    prb = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(prb)
    return p

def test_set_diffraction_from_forward_registers_data():
    """generate_diffraction結果をPtychoに登録できる"""
    p = create_simple_ptycho()
    positions = [(32, 32), (40, 40)]

    diff_list = generate_diffraction(p, positions)
    p.set_diffraction_from_forward(diff_list)

    # scan_posとdiffsが正しく更新される
    assert p.scan_pos == positions
    assert len(p.diffs) == len(positions)
    for d in p.diffs:
        assert d.shape == (16, 16)  # プローブサイズと一致

def test_forward_and_set_diffraction_executes_and_registers():
    """forward_and_set_diffractionがforward実行と登録を行う"""
    p = create_simple_ptycho()
    positions = [(32, 32), (40, 40)]
    p.forward_and_set_diffraction(positions)

    assert p.scan_pos == positions
    assert len(p.diffs) == 2
    assert all(diff.shape == (16, 16) for diff in p.diffs)

def test_forward_and_set_diffraction_append():
    """append=Trueで既存データに追加できる"""
    p = create_simple_ptycho()
    positions1 = [(32, 32)]
    positions2 = [(40, 40)]

    # 1回目
    p.forward_and_set_diffraction(positions1)
    assert len(p.diffs) == 1

    # 追加
    p.forward_and_set_diffraction(positions2, append=True)
    assert len(p.diffs) == 2
    # scan_posが両方の位置を含む
    assert p.scan_pos == positions1 + positions2

def test_clear_then_forward_replaces_data():
    """append=Falseで既存データをクリアして置換"""
    p = create_simple_ptycho()
    positions1 = [(32, 32)]
    positions2 = [(40, 40)]

    p.forward_and_set_diffraction(positions1)
    assert len(p.diffs) == 1

    # append=Falseで置換
    p.forward_and_set_diffraction(positions2, append=False)
    assert len(p.diffs) == 1
    assert p.scan_pos == positions2
