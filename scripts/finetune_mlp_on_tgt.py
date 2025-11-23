# finetune_mlp_on_tgt.py
import argparse, os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mlp_model import MLPRegressor


def ensure_feature_cols(df: pd.DataFrame, numeric_cols):
    """
    和训练时保持一致：
      - 构造 pol_bit
      - 补齐 numeric_cols
    """
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


class TimingDataset(Dataset):
    """
    只用于微调：x,y 用 baseline 的归一化参数 (x_mean/std, y_mean/std)
    """
    def __init__(self, df: pd.DataFrame,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: float, y_std: float,
                 numeric_cols):
        super().__init__()
        self.numeric_cols = numeric_cols
        self.x = df[numeric_cols].values.astype(np.float32)
        self.y = df["delay"].values.astype(np.float32)

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = np.float32(y_mean)
        self.y_std = np.float32(y_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x_norm), torch.tensor(y_norm, dtype=torch.float32)


@torch.no_grad()
def eval_dataset(model, df, x_mean, x_std, y_mean, y_std, numeric_cols, device, name="TGT_L"):
    model.eval()
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

    print(
        f"[Eval {name}] MAE={mae:.4f} ps  MSE={mse:.4f} ps^2  "
        f"RMSE={rmse:.4f} ps  R^2={r2:.6f}  N={len(df)}"
    )
    return mae, mse, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="包含 mlp_ckpt.pt 和 tgt_delay_labeled.csv 的目录")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)  # 小学习率微调
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    # 1) 载入 baseline ckpt
    ckpt_path = os.path.join(args.data_dir, "mlp_ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"[error] baseline ckpt not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    numeric_cols = state["numeric_cols"]
    x_mean = np.array(state["x_mean"], dtype=np.float32)
    x_std = np.array(state["x_std"], dtype=np.float32)
    y_mean = float(state["y_mean"])
    y_std = float(state["y_std"])
    hid_dim = int(state["hid_dim"])
    depth = int(state["depth"])

    # 2) 重建 MLP 并加载参数（相当于从源域预训练模型开始）
    model = MLPRegressor(
        in_dim=len(numeric_cols),
        hid_dim=hid_dim,
        depth=depth,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model"])

    # 3) 加载目标域有标签数据
    tgt_l_csv = os.path.join(args.data_dir, "tgt_delay_labeled.csv")
    if not os.path.exists(tgt_l_csv):
        raise SystemExit(f"[error] tgt_delay_labeled.csv not found: {tgt_l_csv}")

    df_tgt_l = pd.read_csv(tgt_l_csv)
    df_tgt_l = ensure_feature_cols(df_tgt_l, numeric_cols)

    ds_tgt_l = TimingDataset(df_tgt_l, x_mean, x_std, y_mean, y_std, numeric_cols)
    loader_tgt_l = DataLoader(ds_tgt_l, batch_size=args.batch, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    best_rmse = 1e9
    best_state = None

    print("[info] Start fine-tuning on target labeled data...")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for xb, yb in loader_tgt_l:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        avg_loss = total_loss / max(n_samples, 1)
        print(f"[Epoch {ep}] fine-tune MSE (norm space) = {avg_loss:.6f}")

        # 在目标有标签集上评估（物理单位）
        mae, mse, rmse, r2 = eval_dataset(
            model, df_tgt_l, x_mean, x_std, y_mean, y_std, numeric_cols, device, name="TGT_L"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {
                "model": model.state_dict(),
                "numeric_cols": numeric_cols,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean,
                "y_std": y_std,
                "hid_dim": hid_dim,
                "depth": depth,
                "note": "fine-tuned on target labeled",
            }

    # 4) 保存微调后的模型
    out_ckpt = os.path.join(args.data_dir, "mlp_ft_tgt_ckpt.pt")
    torch.save(best_state, out_ckpt)
    print(f"[info] Saved fine-tuned MLP to {out_ckpt}, best TGT_L RMSE={best_rmse:.4f} ps")


if __name__ == "__main__":
    main()
