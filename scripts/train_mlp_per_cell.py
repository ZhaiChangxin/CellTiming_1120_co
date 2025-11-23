# train_mlp_per_cell.py
import argparse, os, json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mlp_model import MLPRegressor


# 我们统一 5 个目标 cell 类型
CELL_TYPES = ["INVX1", "INVX2", "NANDX1", "NORX1", "XORX1"]

# 数值特征列（和之前保持一致）
NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv", "stack_pu", "stack_pd",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit",
]

TARGET_COL = "delay"


def ensure_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    保证 df 中：
      - 有 pol_bit: pol == 'rise' -> 1.0, 其他 0.0
      - 有 NUMERIC_COLS 中所有列（缺的补 0）
      - delay 为 float32
    """
    df = df.copy()

    if "pol_bit" not in df.columns:
        if "pol" in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df["pol_bit"] = 0.0

    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype(np.float32)
    df[TARGET_COL] = df[TARGET_COL].astype(np.float32)
    return df


class TimingDataset(Dataset):
    """
    单 cell_type 回归数据集，内部做标准化:
      x_norm = (x - x_mean) / x_std
      y_norm = (y - y_mean) / y_std
    """
    def __init__(self, df: pd.DataFrame,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: float, y_std: float):
        super().__init__()
        self.df = df.reset_index(drop=True)

        self.x = self.df[NUMERIC_COLS].values.astype(np.float32)
        self.y = self.df[TARGET_COL].values.astype(np.float32)

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = np.float32(y_mean)
        self.y_std = np.float32(y_std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x_norm), torch.tensor(y_norm, dtype=torch.float32)


@torch.no_grad()
def eval_dataset(model, df: pd.DataFrame,
                 x_mean, x_std, y_mean, y_std, device, name=""):
    """
    在指定 df 上评估:
      - 把 preds 从归一化空间反变换回 ps
      - 计算 MAE / MSE / RMSE / R^2
    """
    model.eval()
    xs = df[NUMERIC_COLS].values.astype(np.float32)
    ys = df[TARGET_COL].values.astype(np.float32)

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

    print(
        f"[Eval {name}] MAE={mae:.4f} ps  MSE={mse:.4f} ps^2  "
        f"RMSE={rmse:.4f} ps  R^2={r2:.6f}  N={len(df)}"
    )
    return mae, mse, rmse, r2


def train_for_one_cell(cell_type: str,
                       df_src: pd.DataFrame,
                       df_tgt_l: pd.DataFrame,
                       out_dir: str,
                       epochs: int = 100,
                       batch_size: int = 256,
                       lr: float = 2e-3,
                       hid_dim: int = 128,
                       depth: int = 3,
                       device: str = "cpu"):
    """
    针对一个 cell_type 训练一个 MLP
    - 训练数据: 源域 + 目标有标签 的该 cell 样本
    - 归一化: 用这两部分一起算 mean/std
    - 最优模型: 按在目标有标集上的 RMSE 选
    """

    # 过滤该 cell_type 的样本
    df_src_c = df_src[df_src["cell_type"] == cell_type].copy()
    df_tgt_l_c = df_tgt_l[df_tgt_l["cell_type"] == cell_type].copy()

    if len(df_src_c) == 0 or len(df_tgt_l_c) == 0:
        print(f"[WARN] cell_type={cell_type}: src or tgt_l has 0 samples, skip.")
        return

    print(f"\n========== Train cell_type = {cell_type} ==========")
    print(f"[info] SRC samples     : {len(df_src_c)}")
    print(f"[info] TGT labeled samp: {len(df_tgt_l_c)}")

    # 归一化统计用 src+tg_l 的该 cell
    df_norm = pd.concat([df_src_c, df_tgt_l_c], axis=0).reset_index(drop=True)

    x_all = df_norm[NUMERIC_COLS].astype(np.float32)
    y_all = df_norm[[TARGET_COL]].astype(np.float32)

    x_mean = x_all.mean(0).values
    x_std = x_all.std(0).values + 1e-9
    y_mean = float(y_all.mean().values[0])
    y_std = float(y_all.std().values[0] + 1e-9)

    # 保存 scaler
    scaler_path = os.path.join(out_dir, f"mlp_{cell_type}_scaler.json")
    json.dump(
        {
            "cell_type": cell_type,
            "numeric_cols": NUMERIC_COLS,
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
        },
        open(scaler_path, "w"),
        indent=2
    )
    print(f"[info] saved scaler for {cell_type} to {scaler_path}")

    # Dataset / DataLoader
    df_train = df_norm  # 简单起见：训练集 = src_c + tgt_l_c 全部
    train_ds = TimingDataset(df_train, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # 模型
    model = MLPRegressor(
        in_dim=len(NUMERIC_COLS),
        hid_dim=hid_dim,
        depth=depth,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    best_rmse_tgt = 1e9
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samp = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_samp += xb.size(0)

        avg_loss = total_loss / max(n_samp, 1)
        print(f"[{cell_type}][Epoch {ep}] train MSE (norm) = {avg_loss:.6f}")

        # 用源域/目标有标分别评估
        mae_s, mse_s, rmse_s, r2_s = eval_dataset(
            model, df_src_c, x_mean, x_std, y_mean, y_std, device,
            name=f"{cell_type}-SRC"
        )
        mae_t, mse_t, rmse_t, r2_t = eval_dataset(
            model, df_tgt_l_c, x_mean, x_std, y_mean, y_std, device,
            name=f"{cell_type}-TGT_L"
        )

        # 按目标有标 RMSE 选最优
        if rmse_t < best_rmse_tgt:
            best_rmse_tgt = rmse_t
            best_state = {
                "model": model.state_dict(),
                "cell_type": cell_type,
                "numeric_cols": NUMERIC_COLS,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean,
                "y_std": y_std,
                "hid_dim": hid_dim,
                "depth": depth,
            }

    if best_state is not None:
        ckpt_path = os.path.join(out_dir, f"mlp_{cell_type}_ckpt.pt")
        torch.save(best_state, ckpt_path)
        print(f"[info] Saved best MLP for {cell_type} to {ckpt_path}, best TGT_L RMSE={best_rmse_tgt:.4f} ps")
    else:
        print(f"[WARN] No best_state for {cell_type}, maybe no training was done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="包含 src_delay.csv / tgt_delay_labeled.csv 的目录 (比如 out_dataset)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    tgt_l_csv = os.path.join(args.data_dir, "tgt_delay_labeled.csv")

    if not os.path.exists(src_csv):
        raise SystemExit(f"[error] src_delay.csv not found: {src_csv}")
    if not os.path.exists(tgt_l_csv):
        raise SystemExit(f"[error] tgt_delay_labeled.csv not found: {tgt_l_csv}")

    df_src = pd.read_csv(src_csv)
    df_tgt_l = pd.read_csv(tgt_l_csv)

    df_src = ensure_feature_cols(df_src)
    df_tgt_l = ensure_feature_cols(df_tgt_l)

    # 逐个 cell_type 训练
    for ctype in CELL_TYPES:
        train_for_one_cell(
            ctype,
            df_src,
            df_tgt_l,
            out_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            hid_dim=args.hid,
            depth=args.depth,
            device=device,
        )


if __name__ == "__main__":
    main()
