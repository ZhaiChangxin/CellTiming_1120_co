# eval_mlp_on_splits.py
"""
在目标域带标签数据 (tgt_delay_labeled.csv) 上评估 MLP 模型。

功能：
  - 读取：
      * tgt_delay_labeled.csv
      * mlp_scaler.json
      * split_masks.pt
      * mlp_ckpt.pt
  - 用 scaler 对特征做标准化；
  - 加载 MLP 模型（兼容两种 ckpt 格式）：
      * 新格式：state["model"] 是 net.* 结构（老版 MLP），用 in_dim/hid_dim/depth 建模
      * 旧格式：state 直接是 state_dict
    并支持把 ckpt 里的 net.* key 自动映射到当前 MLPRegressor 的 backbone.* / head.* 上；
  - 在 train/val/test 上输出 MAE / RMSE / MSE。
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from parse_lib import ensure_feature_cols

# 你的工程里 MLPRegressor 在哪里，就按这个导入
try:
    from mlp_model import MLPRegressor
except ImportError:
    from train_mlp import MLPRegressor


# ----------------- 工具函数 -----------------


def load_scaler(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features_from_csv(csv_path: str, scaler_path: str) -> Tuple[np.ndarray, np.ndarray, float, float, list]:
    if not os.path.exists(csv_path):
        raise SystemExit(f"[error] CSV not found: {csv_path}")
    if not os.path.exists(scaler_path):
        raise SystemExit(f"[error] scaler json not found: {scaler_path}")

    print(f"[info] Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    scaler = load_scaler(scaler_path)
    numeric_cols = scaler["numeric_cols"]
    x_mean = np.array(scaler["x_mean"], dtype=np.float32)
    x_std = np.array(scaler["x_std"], dtype=np.float32)
    x_std[x_std == 0] = 1.0

    y_mean = float(scaler.get("y_mean", 0.0))
    y_std = float(scaler.get("y_std", 1.0))
    if y_std == 0:
        y_std = 1.0

    df_feat = ensure_feature_cols(df, numeric_cols)
    X = df_feat[numeric_cols].to_numpy(dtype=np.float32)
    X = (X - x_mean) / x_std

    if "delay" not in df.columns:
        raise SystemExit("[error] CSV 中没有 delay 列，无法评估回归模型。")
    y = df["delay"].to_numpy(dtype=np.float32)

    print(f"[info] num_samples={len(df)}, feature_dim={X.shape[1]}")
    return X, y, y_mean, y_std, numeric_cols


def load_split_masks(path: str, num_samples: int):
    if not os.path.exists(path):
        raise SystemExit(f"[error] split_masks.pt not found: {path}")
    masks = torch.load(path, map_location="cpu")
    train_mask = masks["train_mask"].bool()
    val_mask = masks["val_mask"].bool()
    test_mask = masks["test_mask"].bool()
    print(f"[info] num_labeled from split_masks = {int((train_mask | val_mask | test_mask).sum().item())}")

    if num_samples > 0 and train_mask.numel() != num_samples:
        print(f"[warn] train_mask length ({train_mask.numel()}) != num_samples ({num_samples}), "
              f"请确认 split_masks 与 CSV 是否对应。")
    return train_mask, val_mask, test_mask


def remap_net_to_backbone_head(model_state: dict) -> dict:
    """
    把老 ckpt 里 'net.0.weight' 这类 key 映射到
    当前 MLPRegressor 使用的 'backbone.0.weight' / 'head.weight' 上。

    映射规则（根据报错推出来的结构）：
      net.0.* -> backbone.0.*
      net.2.* -> backbone.2.*
      net.4.* -> backbone.4.*
      net.6.* -> backbone.6.*
      net.8.* -> head.*
    """
    new_state = {}
    mapping = {
        "net.0.": "backbone.0.",
        "net.2.": "backbone.2.",
        "net.4.": "backbone.4.",
        "net.6.": "backbone.6.",
        "net.8.": "head.",
    }

    used = set()

    for k, v in model_state.items():
        mapped = False
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                new_state[new_k] = v
                used.add(old_prefix)
                mapped = True
                break
        if not mapped:
            # 其它 key 保持原样（比如可能有 bn/其它字段）
            new_state[k] = v

    if used:
        print(f"[info] remap net.* -> backbone/head done, used prefixes: {sorted(list(used))}")
    else:
        print("[info] no net.* keys found to remap")

    return new_state


def build_mlp_from_ckpt(state, X_shape, args) -> MLPRegressor:
    """
    根据 ckpt 内容构建 MLP 模型：
      - 新格式：state 是 dict，包含 "model", "in_dim", "hid_dim", "depth"；
      - 旧格式：state 直接是 state_dict（只有权重）。
    再根据 key 名自动做 net.* -> backbone/head 的映射。
    """
    dropout = 0.0  # 评估阶段不用 dropout

    # 新格式 ckpt
    if isinstance(state, dict) and "in_dim" in state and "hid_dim" in state and "depth" in state:
        in_dim = int(state["in_dim"])
        hid_dim = int(state["hid_dim"])
        depth = int(state["depth"])
        raw_state = state["model"]
        print(f"[info] NEW ckpt: in_dim={in_dim}, hid_dim={hid_dim}, depth={depth}")
    else:
        # 旧格式：state 就是 state_dict
        print("[warn] OLD ckpt detected（state 里没有 in_dim/hid_dim/depth），"
              "将使用 X.shape[1] + --hid/--depth 构建模型。")
        in_dim = int(X_shape[1])
        hid_dim = int(args.hid)
        depth = int(args.depth)
        raw_state = state
        print(f"[info] OLD ckpt config: in_dim={in_dim}, hid_dim={hid_dim}, depth={depth}")

    # 先建一个当前版本的 MLPRegressor
    # 注意你的构造函数是 (in_dim, hid_dim, depth, dropout)
    model = MLPRegressor(
        in_dim=in_dim,
        hid_dim=hid_dim,
        depth=depth,
        dropout=dropout,
    )

    # 如果 ckpt 里的 key 里有 "net."，做一次映射
    if any(k.startswith("net.") for k in raw_state.keys()):
        print("[info] detected 'net.' keys in ckpt, remapping to 'backbone'/'head' ...")
        mapped_state = remap_net_to_backbone_head(raw_state)
        model.load_state_dict(mapped_state, strict=False)
    else:
        model.load_state_dict(raw_state, strict=False)

    return model


@torch.no_grad()
def eval_split(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor, name: str):
    mask = mask.bool()
    if mask.sum() == 0:
        print(f"=== {name} ===")
        print("no samples")
        print("")
        return

    p = pred[mask]
    t = true[mask]

    diff = p - t
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5

    print(f"=== {name} ===")
    print(f"MAE : {mae:.4f} ps")
    print(f"RMSE: {rmse:.4f} ps")
    print(f"MSE : {mse:.4f} ps^2")
    print(f"N   : {int(mask.sum().item())}")
    print("")


# ----------------- main -----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv_name", default="tgt_delay_labeled.csv")
    ap.add_argument("--ckpt_name", default="mlp_ckpt.pt")
    ap.add_argument("--split_masks_name", default="split_masks.pt")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="评估设备：cuda 或 cpu",
    )
    # 只在旧格式 ckpt 时需要
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)

    args = ap.parse_args()
    device = torch.device(args.device)

    data_dir = args.data_dir
    csv_path = os.path.join(data_dir, args.csv_name)
    scaler_path = os.path.join(data_dir, "mlp_scaler.json")
    split_path = os.path.join(data_dir, args.split_masks_name)
    ckpt_path = os.path.join(data_dir, args.ckpt_name)

    # 1) 构建特征 / 标签
    X_np, y_np, y_mean, y_std, _ = build_features_from_csv(csv_path, scaler_path)
    num_samples = X_np.shape[0]

    # 2) 加载 split masks
    train_mask, val_mask, test_mask = load_split_masks(split_path, num_samples)

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    # 3) 加载 ckpt 并构建模型
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"[error] MLP ckpt not found: {ckpt_path}")
    print(f"[info] Loading MLP ckpt from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    model = build_mlp_from_ckpt(state, X.shape, args).to(device)
    model.eval()

    # 4) 前向 & 反标准化（如果训练时对 y 做了标准化）
    with torch.no_grad():
        y_pred_norm = model(X)
        # 训练时如果没标准化 y，这里 y_std≈1, y_mean≈0，不会影响数值
        y_pred = y_pred_norm * y_std + y_mean

    # 5) 在 TRAIN / VAL / TEST 上分别打印指标
    eval_split(y_pred, y, train_mask, "TRAIN")
    eval_split(y_pred, y, val_mask, "VAL")
    eval_split(y_pred, y, test_mask, "TEST")


if __name__ == "__main__":
    main()
