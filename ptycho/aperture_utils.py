from ptychoep.backend.backend import np

def circular_aperture(size: int, r: float, dtype=None):
    """
    Generate a circular aperture (used as a probe function).

    This function creates a square 2D array of shape (size, size) with a 
    binary circular mask centered at the origin. The aperture has radius 
    r * size, where r is relative to the array size.

    Args:
        size (int): The height and width of the output array.
        r (float): Aperture radius as a fraction of array size. Must be in (0, 0.5].
        dtype: Data type of the output array. Defaults to complex64.

    Returns:
        xp.ndarray: A (size, size) array with circular mask values.
    """
    xp = np()
    if dtype is None:
        dtype = xp.complex64

    if not (0.0 < r <= 0.5):
        raise ValueError(f"r must be in (0, 0.5], got {r}")

    # Coordinate grid centered at (0, 0)
    y, x = xp.ogrid[-size//2:size//2, -size//2:size//2]
    radius = xp.sqrt(x**2 + y**2)

    # Circular mask
    aperture = xp.zeros((size, size), dtype=dtype)
    aperture[radius <= (r * size)] = 1.0
    return aperture
