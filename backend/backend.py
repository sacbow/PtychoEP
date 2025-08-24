import numpy as _np
try:
    import cupy as _cp
except ImportError:
    _cp = None

_backend = _np  # Default to NumPy

def set_backend(name: str):
    """
    Set the backend to use for numerical computation.

    Parameters
    ----------
    name : str
        Name of the backend. Must be either "numpy" or "cupy".
    
    Raises
    ------
    ImportError
        If "cupy" is specified but CuPy is not installed.
    ValueError
        If an unknown backend name is provided.
    """
    global _backend
    if name == "numpy":
        _backend = _np
    elif name == "cupy":
        if _cp is None:
            raise ImportError("CuPy is not installed.")
        _backend = _cp
    else:
        raise ValueError(f"Unknown backend: {name}")

def np():
    """
    Return the current backend module (NumPy or CuPy).

    Returns
    -------
    module
        The currently active numerical backend.
    """
    return _backend

def is_cupy():
    """
    Check if the current backend is CuPy.

    Returns
    -------
    bool
        True if CuPy is active; False otherwise.
    """
    return _backend.__name__ == "cupy"
