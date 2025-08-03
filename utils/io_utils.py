import os
from skimage.io import imread
from skimage.color import rgb2gray
from .backend import np

def load_image_as_array(filename: str, normalize: bool = True) -> np().ndarray:
    img = imread(filename)
    if img.ndim == 3:  # RGBの場合はグレースケール化
        img = rgb2gray(img)
    img = img.astype(np().float32)
    if normalize:
        max_val = np().max(img)
        if max_val > 0:
            img /= max_val
    return img

def load_data_image(name: str, data_dir: str = None, normalize: bool = True) -> np().ndarray:
    """
    dataフォルダにある標準画像を読み込み。
    """
    if data_dir is None:
        # io_utils.pyから見て../../data に設定
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.abspath(os.path.join(data_dir, name))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return load_image_as_array(path, normalize=normalize)
