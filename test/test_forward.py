import pytest
import numpy as np
from PtychoEP.ptycho.forward import generate_diffraction
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.ptycho.data import DiffractionData
from PtychoEP.backend.backend import set_backend, np as backend_np

@pytest.fixture(autouse=True)
def setup_backend():
    """テストはnumpyバックエンドで固定"""
    set_backend("numpy")

def test_generate_diffraction_basic():
    """基本動作：単純なオブジェクトとプローブから回折像が生成される"""
    # 物体・プローブ設定
    p = Ptycho()
    obj = backend_np().ones((64, 64), dtype=backend_np().complex64)
    probe = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(probe)

    # スキャン位置
    positions = [(32, 32), (40, 40)]

    diffs = generate_diffraction(p, positions)

    # DiffractionDataが返る
    assert all(isinstance(d, DiffractionData) for d in diffs)
    # 戻り値数が位置数と一致
    assert len(diffs) == len(positions)
    # 各回折像のサイズがプローブサイズと一致
    for d in diffs:
        assert d.diffraction.shape == (16, 16)

def test_generate_diffraction_fft_normalization():
    """FFT正規化(norm='ortho')が効いていることを確認"""
    p = Ptycho()
    obj = backend_np().ones((32, 32), dtype=backend_np().complex64)
    probe = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(probe)

    # 中心スキャン位置のみ
    positions = [(16, 16)]
    diffs = generate_diffraction(p, positions)
    diff = diffs[0].diffraction

    # 入力が全て1なら、FFT(ortho)のDC成分は1*sqrt(N^2)=1で一定値
    assert backend_np().allclose(diff[0, 0], 16) 

def test_generate_diffraction_invalid_inputs():
    """オブジェクトやプローブ未設定時に例外が発生"""
    p = Ptycho()
    positions = [(10, 10)]
    # obj未設定
    p.set_probe(backend_np().ones((16, 16), dtype=backend_np().complex64))
    with pytest.raises(ValueError):
        generate_diffraction(p, positions)

    # prb未設定
    p2 = Ptycho()
    p2.set_object(backend_np().ones((32, 32), dtype=backend_np().complex64))
    with pytest.raises(ValueError):
        generate_diffraction(p2, positions)

def test_generate_diffraction_patch_extraction():
    """パッチ切り出しが正しく動作しているか"""
    p = Ptycho()
    obj = backend_np().zeros((64, 64), dtype=backend_np().complex64)
    obj[24:40, 24:40] = 1.0  # 中央に16x16の1パッチ
    probe = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(probe)

    positions = [(32, 32)]
    diffs = generate_diffraction(p, positions)
    diff = diffs[0].diffraction
    # FFTのDC成分は1のパッチ領域に対応 → 非ゼロ値が得られる
    assert backend_np().abs(diff[0, 0]) > 0
