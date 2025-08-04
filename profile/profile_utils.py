import time
import cProfile
import pstats
import io
import cupy as cp

def time_execution(func, backend="numpy"):
    """
    実行時間を計測するユーティリティ関数。
    backend="cupy"ならGPUイベント計測、それ以外はCPU計測。
    """
    if backend == "cupy":
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record()
        func()
        end.record(); end.synchronize()
        elapsed = cp.cuda.get_elapsed_time(start, end) / 1000.0  # 秒に変換
    else:
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
    return elapsed


def profile_execution(func, sort_key="cumulative", limit=30, output_file=None):
    """
    cProfileを用いた詳細プロファイリング。
    
    Args:
        func: プロファイル対象の無引数関数（lambdaやpartial推奨）
        sort_key: ソート基準 ("cumulative", "time", "calls")
        limit: 表示行数
        output_file: 指定時、結果をファイル保存
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
