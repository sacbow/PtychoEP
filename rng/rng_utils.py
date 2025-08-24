from ptychoep.backend.backend import np, is_cupy

def get_rng(seed=None):
    """
    Return a backend-specific random number generator instance.

    Returns
    -------
    rng : numpy.random.Generator or cupy.random.RandomState
        Random number generator initialized with the given seed.
    """
    if is_cupy():
        import cupy as cp
        return cp.random.RandomState(seed)
    else:
        import numpy as _np
        return _np.random.default_rng(seed)


def normal(rng, mean=0.0, var=1.0, size=None, dtype=np().complex64):
    """
    Generate random samples from a normal distribution, using backend-specific RNG.

    Parameters
    ----------
    rng : Generator or RandomState
        Random number generator from get_rng().
    mean : float
        Mean of the distribution.
    var : float
        Variance of the distribution.
    size : int or tuple
        Output shape.
    dtype : data-type
        Desired output dtype (float32, float64, complex64, complex128).

    Returns
    -------
    ndarray
        Sampled array.
    """
    if dtype in (np().float32, np().float64):
        return rng.normal(loc=mean, scale=var**0.5, size=size).astype(dtype)
    elif dtype == np().complex64:
        return (
            rng.normal(loc=mean, scale=(var / 2) ** 0.5, size=size).astype(np().float32)
            + 1j * rng.normal(loc=mean, scale=(var / 2) ** 0.5, size=size).astype(np().float32)
        )
    elif dtype == np().complex128:
        return (
            rng.normal(loc=mean, scale=(var / 2) ** 0.5, size=size).astype(np().float64)
            + 1j * rng.normal(loc=mean, scale=(var / 2) ** 0.5, size=size).astype(np().float64)
        )
    else:
        raise ValueError("Unsupported dtype.")


def uniform(rng, low=0.0, high=1.0, size=None, dtype=np().float32):
    """
    Generate samples from a uniform distribution using backend-specific RNG.

    Parameters
    ----------
    rng : Generator or RandomState
        Random number generator from get_rng().
    low : float
        Lower bound of the distribution.
    high : float
        Upper bound of the distribution.
    size : int or tuple
        Output shape.
    dtype : data-type
        Desired output dtype.

    Returns
    -------
    ndarray
        Sampled array.
    """
    vals = rng.uniform(low=low, high=high, size=size)
    return vals.astype(dtype)


def randint(rng, low: int, high: int, size=None):
    """
    Generate random integers using backend-specific RNG.

    Parameters
    ----------
    rng : Generator or RandomState
        Random number generator from get_rng().
    low : int
        Lower bound (inclusive).
    high : int
        Upper bound (exclusive).
    size : int or tuple, optional
        Output shape. If None, returns a scalar.

    Returns
    -------
    int or list of int
        Sampled integer(s).
    """
    if is_cupy():
        vals = rng.randint(low, high, size=size)
    else:
        vals = rng.integers(low, high, size=size)
    if size is None:
        return int(vals)
    else:
        return [int(v) for v in vals]


def poisson(rng, lam, size=None):
    """
    Generate Poisson-distributed random samples using backend-specific RNG.

    Parameters
    ----------
    rng : Generator or RandomState
        Random number generator from get_rng().
    lam : float or array-like
        Expected value of the distribution.
    size : int or tuple, optional
        Output shape.

    Returns
    -------
    ndarray
        Sampled array.
    """
    return rng.poisson(lam=lam, size=size)
