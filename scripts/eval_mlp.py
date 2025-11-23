# eval_mlp.py
import argparse
import os

import numpy as np
import pandas as pd
import torch

from mlp_model import MLPRegressor


def ensure_feature_cols(df: pd.DataFrame, numeric_cols):
    """
    和 train_mlp.py 里的逻辑保持一致：
      - 由 pol 构造 pol_bit
      - 由 cell_type 构造 cell_type_* one-hot
      - 缺的列补 0
    """
    df = df.copy()

    # ---------- pol_bit ----------
    if "pol_bit" not in df.columns:
        if "pol" in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df["pol_bit"] = 0.0

    # ---------- cell_type one-hot ----------
    if "cell_type" in df.columns:
        ct_series = df["cell_type"].astype(str)
        for col in numeric_cols:
            if col.startswith("cell_type_"):
                ct_name = col[len("cell_type_") :]
                df[col] = (ct_series == ct_name).astype(np.float32)

    # ---------- 缺的列补 0 ----------
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float32)
    if "delay" in df.columns:
        df["delay"] = df["delay"].astype(np.float32)
    return df


@torch.no_grad()
def eval_on_csv(ckpt_path: str, csv_path: str, device: str = "cpu"):
    print(f"[info] Loading ckpt from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # 从 ckpt 中恢复特征配置和归一化参数
    numeric_cols = state["numeric_cols"]
    x_mean = np.array(state["x_mean"], dtype=np.float32)
    x_std = np.array(state["x_std"], dtype=np.float32)
    y_mean = float(state["y_mean"])
    y_std = float(state["y_std"])
    hid_dim = int(state["hid_dim"])
    depth = int(state["depth"])

    # 建 MLP 模型并加载参数
    model = MLPRegressor(
        in_dim=len(numeric_cols),
        hid_dim=hid_dim,
        depth=depth,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # 读待评估数据
    df = pd.read_csv(csv_path)
    df = ensure_feature_cols(df, numeric_cols)

    xs = df[numeric_cols].values.astype(np.float32)
    ys = df["delay"].values.astype(np.float32)

    xs = (xs - x_mean) / x_std
    xs_t = torch.from_numpy(xs).to(device)
    ys_t = torch.from_numpy(ys).to(device)

    # 预测并反标准化
    preds_norm = model(xs_t)
    preds = preds_norm * y_std + y_mean

    diff = preds - ys_t
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5

    # 简单算个 R^2
    y_bar = ys_t.mean().item()
    sst = torch.sum((ys_t - y_bar) ** 2).item()
    sse = torch.sum(diff ** 2).item()
    r2 = 1.0 - sse / (sst + 1e-12)

    print(f"=== MLP Evaluation on {os.path.basename(csv_path)} ===")
    print(f"MAE : {mae:.4f} ps")
    print(f"MSE : {mse:.4f} ps^2")
    print(f"RMSE: {rmse:.4f} ps")
    print(f"R^2 : {r2:.6f}")
    print(f"N   : {len(df)}")

    return mae, mse, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="包含 ckpt 和 csv 的目录")
    ap.add_argument("--csv_name", default="tgt_delay.csv",
                    help="要评估的 csv 文件名")
    ap.add_argument("--ckpt_name", default="mlp_ckpt.pt",
                    help="要使用的 ckpt 文件名")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_path = os.path.join(args.data_dir, args.ckpt_name)
    csv_path = os.path.join(args.data_dir, args.csv_name)

    if not os.path.exists(ckpt_path):
        raise SystemExit(f"[error] ckpt not found: {ckpt_path}")
    if not os.path.exists(csv_path):
        raise SystemExit(f"[error] csv not found: {csv_path}")

    eval_on_csv(ckpt_path, csv_path, device=args.device)


if __name__ == "__main__":
    main()
