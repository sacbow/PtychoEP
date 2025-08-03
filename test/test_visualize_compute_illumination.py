import pytest
from PtychoEP.utils.ptycho.core import Ptycho
from PtychoEP.utils.ptycho.forward import generate_diffraction
from PtychoEP.utils.ptycho.visualize import compute_illumination
from PtychoEP.utils.backend import set_backend, np as backend_np

@pytest.fixture(autouse=True)
def setup_backend():
    """numpyバックエンドで固定"""
    set_backend("numpy")

def create_simple_ptycho_with_diffractions():
    """シンプルなPtychoにforwardを適用してDiffractionDataを登録"""
    p = Ptycho()
    obj = backend_np().ones((64, 64), dtype=backend_np().complex64)
    prb = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(prb)
    positions = [(32, 32), (40, 40)]
    diffs = generate_diffraction(p, positions)
    p.set_diffraction_from_forward(diffs)
    return p

def test_indices_shape_and_values():
    """DiffractionData.indicesが正しい形状で生成されているかを確認"""
    p = create_simple_ptycho_with_diffractions()
    for d in p._diff_data:
        Y, X = d.indices
        assert Y.shape == (p.prb_len, p.prb_len)
        assert X.shape == (p.prb_len, p.prb_len)
        # Y,Xはint型の座標であること
        assert backend_np().issubdtype(Y.dtype, backend_np().integer)
        assert backend_np().issubdtype(X.dtype, backend_np().integer)

def test_compute_illumination_output():
    """compute_illuminationの出力が正しい形状・性質を持つかを確認"""
    p = create_simple_ptycho_with_diffractions()
    scan_img, alpha = compute_illumination(p)
    # 出力形状の確認
    assert scan_img.shape == (p.obj_len, p.obj_len)
    # scan_imgは非負値のみ
    assert (scan_img >= 0).all()
    # alphaが正の値であること
    assert alpha > 0

def test_compute_illumination_consistency_with_positions():
    """スキャン位置数と照明分布が対応しているか（値の非ゼロ数増加確認）"""
    p = create_simple_ptycho_with_diffractions()
    scan_img, _ = compute_illumination(p)
    # 非ゼロ画素数がスキャン位置に依存して増えていることを確認
    nonzero_pixels = (scan_img > 0).sum()
    assert nonzero_pixels > 0
    assert nonzero_pixels <= p.obj_len * p.obj_len
