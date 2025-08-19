import time
import cProfile
import pstats
import io
import cupy as cp

def time_execution(func, backend="numpy"):
    """
    Measure execution time of a function.

    Uses CPU wall-clock timer by default.
    If backend="cupy", uses CUDA event timing on GPU.

    Parameters
    ----------
    func : callable
        A function with no arguments to be executed and timed.
    backend : str
        "numpy" (default) for CPU timing, "cupy" for GPU timing.

    Returns
    -------
    float
        Elapsed time in seconds.
    """
    if backend == "cupy":
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record()
        func()
        end.record(); end.synchronize()
        elapsed = cp.cuda.get_elapsed_time(start, end) / 1000.0  # Convert ms to sec
    else:
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
    return elapsed


def profile_execution(func, sort_key="cumulative", limit=30, output_file=None):
    """
    Perform detailed profiling using cProfile.

    Useful for inspecting performance bottlenecks in Python functions.

    Parameters
    ----------
    func : callable
        A no-argument function to profile (e.g., lambda or functools.partial).
    sort_key : str
        Sorting key for the report ("cumulative", "time", or "calls").
    limit : int
        Maximum number of lines to print in the report.
    output_file : str or None
        If specified, the profiling result will be saved to this file.
        Otherwise, the result is printed to stdout.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats(sort_key)
    stats.print_stats(limit)
    
    result_str = s.getvalue()
    if output_file:
        with open(output_file, "w") as f:
            f.write(result_str)
    else:
        print(result_str)
