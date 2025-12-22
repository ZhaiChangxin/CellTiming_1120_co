# bucket_utils.py
"""
一些和 delay 分布相关的辅助函数：
  - 按 delay (ps) 做分桶（quantile bucket）
  - 计算每个样本的长尾权重（尾部 bucket 权重大）
  - 打印每个 bucket 的 MAE/RMSE 等统计，方便 debug

MLP / HGAT 都可以共用。
"""

from typing import Tuple, Dict, Optional, Sequence

import numpy as np


def make_delay_buckets(
    y_ps: np.ndarray,
    num_buckets: int = 5,
    method: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 delay（ps）做分桶，返回：
      bucket_ids: 每个样本属于哪个 bucket，shape = (N,)
      edges    : bucket 的边界数组，shape = (num_buckets + 1,)

    这里默认用 quantile（分位数）来切分，适合长尾分布。
    """
    y_ps = np.asarray(y_ps, dtype=np.float64)
    mask = np.isfinite(y_ps)
    y_valid = y_ps[mask]
    if y_valid.size == 0:
        raise ValueError("make_delay_buckets: no valid y values")

    if method == "quantile":
        qs = np.linspace(0.0, 1.0, num_buckets + 1)
        edges = np.quantile(y_valid, qs)
    else:
        raise ValueError(f"unknown bucket method: {method}")

    # 为了方便处理，首尾扩一丢丢
    edges[0] = edges[0] - 1e-6
    edges[-1] = edges[-1] + 1e-6

    # bucket_id: 0..(num_buckets-1)
    bucket_ids = np.digitize(y_ps, edges[1:-1], right=True)
    return bucket_ids.astype(int), edges.astype(float)


def assign_buckets_with_edges(
    y_ps: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    使用已有的 edges 对新的 y_ps 做分桶（保证 train/val/test 用的是同一套 bucket 边界）。
    """
    y_ps = np.asarray(y_ps, dtype=np.float64)
    bucket_ids = np.digitize(y_ps, edges[1:-1], right=True)
    return bucket_ids.astype(int)


def compute_bucket_weights(
    bucket_ids: np.ndarray,
    min_weight: float = 1.0,
    max_weight: float = 5.0,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    根据 bucket 中的样本数量来算长尾权重：
      - 频率越低的 bucket 权重越大；
      - 权重在 [min_weight, max_weight] 之间裁剪。

    返回：
      sample_weights: shape = (N,)
      bucket_weight_map: {bucket_id: weight}
    """
    ids = np.asarray(bucket_ids, dtype=int)
    uniq, counts = np.unique(ids, return_counts=True)

    # freq 越小，inv_freq 越大
    max_count = counts.max()
    inv_freq = max_count / counts.astype(float)  # 比如 count 小时会变大
    # 标准化一下，让平均权重大约在 1 左右
    inv_freq = inv_freq / inv_freq.mean()

    raw_weights = np.clip(inv_freq, min_weight, max_weight)

    bucket_weight_map = {int(b): float(w) for b, w in zip(uniq, raw_weights)}
    sample_weights = np.array([bucket_weight_map[int(b)] for b in ids], dtype=np.float32)
    return sample_weights, bucket_weight_map


def print_bucket_debug(
    y_true_ps: Sequence[float],
    y_pred_ps: Sequence[float],
    bucket_ids: Sequence[int],
    edges: np.ndarray,
    name: str = "TGT",
) -> None:
    """
    按 bucket 打印：
      - N
      - MAE / RMSE
      - y_true 均值

    方便观察长尾 bucket 的误差情况。
    """
    y_true = np.asarray(y_true_ps, dtype=np.float64)
    y_pred = np.asarray(y_pred_ps, dtype=np.float64)
    b = np.asarray(bucket_ids, dtype=int)
    edges = np.asarray(edges, dtype=np.float64)

    assert y_true.shape == y_pred.shape == b.shape

    print(f"==== Bucket debug for {name} ====")
    for bucket_id in sorted(np.unique(b)):
        mask = b == bucket_id
        n = int(mask.sum())
        if n == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        diff = yp - yt
        mae = np.mean(np.abs(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mean_y = float(np.mean(yt))
        lo = edges[bucket_id]
        hi = edges[bucket_id + 1]
        print(
            f"[Bucket {bucket_id}] "
            f"range=[{lo:.2f}, {hi:.2f}] ps, "
            f"N={n}, mean_y={mean_y:.2f} ps, MAE={mae:.3f}, RMSE={rmse:.3f}"
        )
    print("==== End bucket debug ====")
