# eval_mlp_per_cell.py
import argparse, os

import numpy as np
import pandas as pd
import torch

from mlp_model import MLPRegressor


CELL_TYPES = ["INVX1", "INVX2", "NANDX1", "NORX1", "XORX1"]


def ensure_feature_cols(df: pd.DataFrame, numeric_cols):
    df = df.copy()

    if "pol_bit" not in df.columns:
        if "pol" in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df["pol_bit"] = 0.0

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float32)
    df["delay"] = df["delay"].astype(np.float32)
    return df


@torch.no_grad()
def eval_one_cell(data_dir, cell_type, csv_name, device="cpu"):
    """
    对某个 cell_type 的样本进行评估：
      - 加载对应的 ckpt & scaler
      - 只在 csv 中 cell_type == 该类型 的行上评估
    """
    ckpt_path = os.path.join(data_dir, f"mlp_{cell_type}_ckpt.pt")
    scaler_path = os.path.join(data_dir, f"mlp_{cell_type}_scaler.json")
    csv_path = os.path.join(data_dir, csv_name)

    if not os.path.exists(ckpt_path) or not os.path.exists(scaler_path):
        print(f"[WARN] ckpt or scaler for {cell_type} not found, skip.")
        return None

    state = torch.load(ckpt_path, map_location=device)
    import json
    scaler = json.load(open(scaler_path, "r", encoding="utf-8"))

    numeric_cols = scaler["numeric_cols"]
    x_mean = np.array(scaler["x_mean"], dtype=np.float32)
    x_std = np.array(scaler["x_std"], dtype=np.float32)
    y_mean = float(scaler["y_mean"])
    y_std = float(scaler["y_std"])
    hid_dim = int(state["hid_dim"])
    depth = int(state["depth"])

    model = MLPRegressor(
        in_dim=len(numeric_cols),
        hid_dim=hid_dim,
        depth=depth,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    df = pd.read_csv(csv_path)
    # 只保留该 cell_type
    df = df[df["cell_type"] == cell_type].copy()
    if len(df) == 0:
        print(f"[INFO] No samples with cell_type={cell_type} in {csv_name}, skip.")
        return None

    df = ensure_feature_cols(df, numeric_cols)

    xs = df[numeric_cols].values.astype(np.float32)
    ys = df["delay"].values.astype(np.float32)

    xs = (xs - x_mean) / x_std
    xs_t = torch.from_numpy(xs).to(device)
    ys_t = torch.from_numpy(ys).to(device)

    preds_norm = model(xs_t)
    preds = preds_norm * y_std + y_mean

    diff = preds - ys_t
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5

    y_bar = ys_t.mean().item()
    sst = torch.sum((ys_t - y_bar) ** 2).item()
    sse = torch.sum(diff ** 2).item()
    r2 = 1.0 - sse / (sst + 1e-12)

    print(f"=== Eval {cell_type} on {csv_name} ===")
    print(f"MAE : {mae:.4f} ps")
    print(f"MSE : {mse:.4f} ps^2")
    print(f"RMSE: {rmse:.4f} ps")
    print(f"R^2 : {r2:.6f}")
    print(f"N   : {len(df)}")

    # 返回整体误差和 N，方便之后汇总
    return {
        "cell_type": cell_type,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "n": len(df),
        "sse": sse,
        "sst": sst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="包含 mlp_*_ckpt.pt 和目标 csv 的目录")
    ap.add_argument("--csv_name", default="tgt_delay.csv",
                    help="要评估的 csv（默认 tgt_delay.csv）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    results = []
    for ctype in CELL_TYPES:
        r = eval_one_cell(args.data_dir, ctype, args.csv_name, device=device)
        if r is not None:
            results.append(r)

    # 汇总整体指标（按样本数加权）
    if not results:
        print("[WARN] No cell_type evaluated, nothing to summarize.")
        return

    total_n = sum(r["n"] for r in results)
    weighted_mae = sum(r["mae"] * r["n"] for r in results) / total_n
    weighted_mse = sum(r["mse"] * r["n"] for r in results) / total_n
    weighted_rmse = weighted_mse ** 0.5

    # 全局 R^2 = 1 - (ΣSSE)/(ΣSST)
    total_sse = sum(r["sse"] for r in results)
    total_sst = sum(r["sst"] for r in results)
    global_r2 = 1.0 - total_sse / (total_sst + 1e-12)

    print("\n=== Overall (all 5 cell_types, weighted by N) ===")
    print(f"MAE : {weighted_mae:.4f} ps")
    print(f"MSE : {weighted_mse:.4f} ps^2")
    print(f"RMSE: {weighted_rmse:.4f} ps")
    print(f"R^2 : {global_r2:.6f}")
    print(f"N   : {total_n}")


if __name__ == "__main__":
    main()
