import pytest
import numpy as np
from PtychoEP.ptycho.core import Ptycho
from PtychoEP.ptycho.data import DiffractionData
from PtychoEP.backend.backend import set_backend, np as backend_np

@pytest.fixture(autouse=True)
def setup_backend():
    """テスト実行時に常にnumpyバックエンドを利用"""
    set_backend("numpy")

def test_initialization():
    """初期化後、空の状態であることを確認"""
    p = Ptycho()
    assert p.obj is None
    assert p.prb is None
    assert p.obj_len is None
    assert p.prb_len is None
    assert p.scan_pos == []
    assert p.diffs == []

def test_set_object_and_probe():
    """set_objectとset_probeが正常に動作し、サイズが記録される"""
    p = Ptycho()
    obj = backend_np().zeros((64, 64), dtype=backend_np().complex64)
    probe = backend_np().zeros((16, 16), dtype=backend_np().complex64)

    p.set_object(obj)
    p.set_probe(probe)

    assert p.obj.shape == (64, 64)
    assert p.obj_len == 64
    assert p.prb.shape == (16, 16)
    assert p.prb_len == 16

def test_set_object_invalid_shape():
    """set_objectに非正方行列を与えた場合にエラーが出る"""
    p = Ptycho()
    bad_obj = backend_np().zeros((64, 32), dtype=backend_np().complex64)
    with pytest.raises(ValueError):
        p.set_object(bad_obj)

def test_set_probe_invalid_shape():
    """set_probeに非正方行列を与えた場合にエラーが出る"""
    p = Ptycho()
    bad_probe = backend_np().zeros((32, 16), dtype=backend_np().complex64)
    with pytest.raises(ValueError):
        p.set_probe(bad_probe)

def test_add_diffraction_data_and_properties():
    """DiffractionData追加とscan_pos/diffsプロパティの挙動を確認"""
    p = Ptycho()
    diff_arr1 = backend_np().ones((8, 8), dtype=backend_np().complex64)
    diff_arr2 = backend_np().full((8, 8), 2+2j, dtype=backend_np().complex64)

    d1 = DiffractionData(position=(10, 10), diffraction=diff_arr1)
    d2 = DiffractionData(position=(20, 20), diffraction=diff_arr2)

    # 追加
    p.add_diffraction_data(d1)
    p.add_diffraction_data(d2)

    # scan_posとdiffsが正しいか
    assert p.scan_pos == [(10, 10), (20, 20)]
    assert len(p.diffs) == 2
    assert backend_np().allclose(p.diffs[0], diff_arr1)
    assert backend_np().allclose(p.diffs[1], diff_arr2)

def test_add_diffraction_data_list_and_clear():
    """複数追加とクリア機能の動作を確認"""
    p = Ptycho()
    diff_list = [
        DiffractionData(position=(0, 0), diffraction=backend_np().zeros((4, 4), dtype=backend_np().complex64)),
        DiffractionData(position=(5, 5), diffraction=backend_np().ones((4, 4), dtype=backend_np().complex64)),
    ]

    p.add_diffraction_data_list(diff_list)
    assert len(p.scan_pos) == 2

    p.clear_diffraction_data()
    assert p.scan_pos == []
    assert p.diffs == []

def test_add_diffraction_data_invalid_type():
    """add_diffraction_dataに不正な型を渡した場合にエラーが出る"""
    p = Ptycho()
    with pytest.raises(TypeError):
        p.add_diffraction_data("not a DiffractionData")  # type: ignore
