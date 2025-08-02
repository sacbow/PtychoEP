#imports
import numpy as np
from numpy import exp, pi
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc
rc('text', usetex=True)
from PIL import Image
from matplotlib.colors import Normalize

#load two images and output complex 2d-array
def load_complex_image(path_abs, path_phase):
    image_abs = np.array(Image.open(path_abs), dtype = "float")/256
    image_phase = np.array(Image.open(path_phase), dtype = "float")/256
    return image_abs * exp(0.5 * pi* 1j * image_phase)

#show images
def show_complex_image(img, img_name):
    fig, ax = plt.subplots(1,2, figsize=(6, 3))
    im1 = ax[0].imshow(np.abs(img), cmap = "gray", norm = Normalize(vmin = 0, vmax = 1))
    ax[0].set_title(img_name + " (Amplitude)", fontsize = 15)
    ax[0].axis("off")
    im2 = ax[1].imshow(np.angle(img), cmap="gray")
    ax[1].set_title(img_name + " (Phase)", fontsize = 15)
    ax[1].axis("off")
    plt.show()

