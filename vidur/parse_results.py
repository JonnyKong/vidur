import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class PerfStats:
    throughput: float   # Requests per second
    ttft_mean: float
    ttft_p99: float
    tbt_mean: float
    tbt_p99: float
    power_w: float
    freq_mhz_mean: float
    freq_mhz_p10: float
    freq_mhz_p50: float
    freq_mhz_p90: float
    mem_util_mean: float
    mem_util_p10: float
    mem_util_p50: float
    mem_util_p90: float
    batch_size_mean: float
    running_queue_len_mean: float
    waiting_queue_len_mean: float
    expr_duration_s: float


def extract_steady_region(
        df_stats: pd.DataFrame,
        df_requests: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts the steady region of a vLLM serving session based on `running_queue_len`.
    Also extracts the corresponding steady region from `df_requests`.

    Parameters:
    - df_stats (pd.DataFrame): DataFrame containing system stats, ordered in time.
    - df_requests (pd.DataFrame): DataFrame containing request timestamps.
    """
    if "running_queue_len" not in df_stats:
        raise ValueError("Column 'running_queue_len' not found in df_stats")
    if "ts" not in df_requests:
        raise ValueError("Column 'ts' not found in df_requests")

    max_val = df_stats["running_queue_len"].max()
    threshold = 0.5 * max_val

    # Find first and last occurrence of reaching the threshold
    first_idx = df_stats[df_stats["running_queue_len"] >= threshold].index.min()
    last_idx = df_stats[df_stats["running_queue_len"] >= threshold].index.max()

    # Extract the steady region for df_stats
    steady_region_stats = df_stats.loc[first_idx:last_idx]

    # Extract the corresponding steady region for df_requests
    start_time = df_stats.loc[first_idx, "ts"]
    end_time = df_stats.loc[last_idx, "ts"]
    steady_region_requests = df_requests[(df_requests["ts"] >= start_time)
                                         & (df_requests["ts"] <= end_time)]

    return steady_region_stats, steady_region_requests


def load_trace_into_df(trace_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(trace_path) as f:
        trace = json.load(f)['traceEvents']
    df_stats = []
    df_batches = []
    df_requests = []
    for t in trace:
        if t['name'] == 'stats':
            df_stats.append({
                'ts': t['ts'],
                **t['args'],
            })
        elif t['ph'] == 'X':
            df_batches.append({
                'ts': t['ts'],
                **t['args'],
            })
        elif t['name'] == 'request_end':
            df_requests.append({
                'ts': t['ts'],
                **t['args'],
            })
    df_stats = pd.DataFrame(df_stats)
    df_batches = pd.DataFrame(df_batches)
    df_requests = pd.DataFrame(df_requests)

    # Merge df_batches into df_stats, since they have one-to-one correspondence
    assert len(df_stats) == len(df_batches)
    assert df_stats.ts.is_monotonic_increasing
    assert df_batches.ts.is_monotonic_increasing
    df_batches = df_batches.drop('ts', axis=1)
    df_stats = df_stats.join(df_batches)

    df_stats, df_requests = extract_steady_region(df_stats, df_requests)
    return df_stats, df_requests


def calc_perf_stats(df_stats: pd.DataFrame, df_requests: pd.DataFrame) -> PerfStats:
    throughput = (len(df_requests) - 1) / (df_requests.ts.max() - df_requests.ts.min()) * 1e6
    ttft_arr = (df_requests.prefill_completed_at - df_requests.arrived_at).to_numpy()
    tbt_arr = df_stats.ts.diff().iloc[1:].to_numpy() / 1e6

    # Energy
    total_busy_energy = (df_stats.last_batch_busy_power * df_stats.last_batch_busy_duration).sum()
    total_idle_energy = (df_stats.last_batch_idle_power * df_stats.last_batch_idle_duration).sum()
    total_busy_duration = df_stats.last_batch_busy_duration.sum()
    total_idle_duration = df_stats.last_batch_idle_duration.sum()
    power_w = (total_busy_energy + total_idle_energy) / (total_busy_duration + total_idle_duration)

    return PerfStats(
        throughput=throughput,
        ttft_mean=float(np.mean(ttft_arr)),
        ttft_p99=float(percentile_or_nan(ttft_arr, q=99)),
        tbt_mean=float(np.mean(tbt_arr)),
        tbt_p99=float(percentile_or_nan(tbt_arr, q=99)),
        power_w=power_w,
        freq_mhz_mean=df_stats.freq.mean(),
        freq_mhz_p10=df_stats.freq.quantile(q=0.1),
        freq_mhz_p50=df_stats.freq.quantile(q=0.5),
        freq_mhz_p90=df_stats.freq.quantile(q=0.9),
        mem_util_mean=df_stats.memory_usage_percent.mean(),
        mem_util_p10=df_stats.memory_usage_percent.quantile(q=0.1),
        mem_util_p50=df_stats.memory_usage_percent.quantile(q=0.5),
        mem_util_p90=df_stats.memory_usage_percent.quantile(q=0.9),
        batch_size_mean=df_stats.batch_size.mean(),
        running_queue_len_mean=df_stats.running_queue_len.mean(),
        waiting_queue_len_mean=df_stats.waiting_queue_len.mean(),
        expr_duration_s=(df_stats.ts.max() - df_stats.ts.min()) / 1e6,
    )


def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan


if __name__ == '__main__':
    if len(sys.argv) > 1:
        expr_root = Path(sys.argv[1])
    else:
        expr_root = Path('/export2/home/kong102/vidur/simulator_output')
    df = []

    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        trace_path = expr_dir / 'chrome_trace.json'

        try:
            df_stats, df_requests = load_trace_into_df(trace_path)
            s = calc_perf_stats(df_stats, df_requests)
        except FileNotFoundError:
            print(f'WARNING: log not found, skipping: {trace_path}')
            continue
        except AssertionError:
            print(f'WARNING: error parsing log, skipping: {trace_path}')
            continue

        df.append({
            'expr_dir': expr_dir.name,
            **asdict(s),
        })
    try:
        pd.DataFrame(df).to_csv(expr_root / 'metrics.csv', index=False)
    except PermissionError:
        save_path = Path.home() / 'metrics.csv'
        pd.DataFrame(df).to_csv(save_path, index=False)
        print(f'No permission to save to expr_dir. Saved to: {save_path}')
