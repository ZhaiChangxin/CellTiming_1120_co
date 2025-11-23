# train_cmd_mlp.py
import argparse, os, json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from cmd_mlp_model import CMDMLPRegressor


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


def cmd_loss(x: torch.Tensor, y: torch.Tensor, n_moments: int = 5) -> torch.Tensor:
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device)

    mx = x.mean(dim=0, keepdim=True)
    my = y.mean(dim=0, keepdim=True)
    loss = torch.norm(mx - my, p=2)

    for k in range(2, n_moments + 1):
        cx = (x - mx) ** k
        cy = (y - my) ** k
        mxk = cx.mean(dim=0)
        myk = cy.mean(dim=0)
        loss = loss + torch.norm(mxk - myk, p=2)

    return loss


@torch.no_grad()
def eval_dataset(model, df: pd.DataFrame,
                 x_mean, x_std, y_mean, y_std, device, name=""):
    model.eval()
    xs = df[NUMERIC_COLS].values.astype(np.float32)
    ys = df[TARGET_COL].values.astype(np.float32)

    xs = (xs - x_mean) / x_std
    xs_t = torch.from_numpy(xs).to(device)
    ys_t = torch.from_numpy(ys).to(device)

    preds_norm, _ = model(xs_t, return_feat=True)
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
                    help="包含 src_delay.csv / tgt_delay_labeled.csv / tgt_delay_unlabeled.csv 的目录")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lambda_cmd", type=float, default=1.0)
    ap.add_argument("--lambda_tgt", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    # ---------- 1) 读数据 ----------
    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    tgt_l_csv = os.path.join(args.data_dir, "tgt_delay_labeled.csv")
    tgt_u_csv = os.path.join(args.data_dir, "tgt_delay_unlabeled.csv")

    df_src = pd.read_csv(src_csv)
    df_tgt_l = pd.read_csv(tgt_l_csv)
    df_tgt_u = pd.read_csv(tgt_u_csv)

    df_src = ensure_feature_cols(df_src)
    df_tgt_l = ensure_feature_cols(df_tgt_l)
    df_tgt_u = ensure_feature_cols(df_tgt_u)

    # 归一化用 src + tgt_l
    df_norm = pd.concat([df_src, df_tgt_l], axis=0).reset_index(drop=True)

    x_all = df_norm[NUMERIC_COLS].astype(np.float32)
    y_all = df_norm[[TARGET_COL]].astype(np.float32)

    x_mean = x_all.mean(0).values
    x_std = x_all.std(0).values + 1e-9
    y_mean = float(y_all.mean().values[0])
    y_std = float(y_all.std().values[0] + 1e-9)

    scaler_path = os.path.join(args.data_dir, "cmd_mlp_scaler.json")
    json.dump(
        {
            "numeric_cols": NUMERIC_COLS,
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
        },
        open(scaler_path, "w"),
        indent=2
    )
    print(f"[info] saved scaler to {scaler_path}")

    # ---------- 2) Dataloader（关键：drop_last=False） ----------
    ds_src = TimingDataset(df_src, x_mean, x_std, y_mean, y_std)
    ds_tgt_l = TimingDataset(df_tgt_l, x_mean, x_std, y_mean, y_std)
    ds_tgt_u = TimingDataset(df_tgt_u, x_mean, x_std, y_mean, y_std)

    loader_src = DataLoader(ds_src, batch_size=args.batch, shuffle=True, drop_last=False)
    loader_tgt_l = DataLoader(ds_tgt_l, batch_size=args.batch, shuffle=True, drop_last=False)
    loader_tgt_u = DataLoader(ds_tgt_u, batch_size=args.batch, shuffle=True, drop_last=False)

    # ---------- 3) 模型 ----------
    model = CMDMLPRegressor(
        in_dim=len(NUMERIC_COLS),
        hid_dim=args.hid,
        depth=args.depth,
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    best_rmse_tgt = 1e9
    best_state = None

    # ---------- 4) 训练 ----------
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0

        it_src = iter(loader_src)
        it_tgt_l = iter(loader_tgt_l)
        it_tgt_u = iter(loader_tgt_u)

        for _ in range(len(loader_src)):
            try:
                xb_s, yb_s = next(it_src)
            except StopIteration:
                it_src = iter(loader_src)
                xb_s, yb_s = next(it_src)

            try:
                xb_tl, yb_tl = next(it_tgt_l)
            except StopIteration:
                it_tgt_l = iter(loader_tgt_l)
                xb_tl, yb_tl = next(it_tgt_l)

            try:
                xb_tu, yb_tu = next(it_tgt_u)
            except StopIteration:
                it_tgt_u = iter(loader_tgt_u)
                xb_tu, yb_tu = next(it_tgt_u)

            xb_s, yb_s = xb_s.to(device), yb_s.to(device)
            xb_tl, yb_tl = xb_tl.to(device), yb_tl.to(device)
            xb_tu = xb_tu.to(device)

            yhat_s, feat_s = model(xb_s, return_feat=True)
            loss_src = mse_loss(yhat_s, yb_s)

            yhat_tl, feat_tl = model(xb_tl, return_feat=True)
            loss_tgt = mse_loss(yhat_tl, yb_tl)

            _, feat_tu = model(xb_tu, return_feat=True)
            feat_t = torch.cat([feat_tl, feat_tu], dim=0)

            loss_cmd = cmd_loss(feat_s, feat_t, n_moments=5)

            loss = loss_src + args.lambda_tgt * loss_tgt + args.lambda_cmd * loss_cmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"[Epoch {ep}] train total loss = {avg_loss:.6f}")

        # 评估
        mae_s, mse_s, rmse_s, r2_s = eval_dataset(
            model, df_src, x_mean, x_std, y_mean, y_std, device, name="SRC"
        )
        mae_t, mse_t, rmse_t, r2_t = eval_dataset(
            model, df_tgt_l, x_mean, x_std, y_mean, y_std, device, name="TGT_L"
        )

        if rmse_t < best_rmse_tgt:
            best_rmse_tgt = rmse_t
            best_state = {
                "model": model.state_dict(),
                "numeric_cols": NUMERIC_COLS,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean,
                "y_std": y_std,
                "hid_dim": args.hid,
                "depth": args.depth,
                "lambda_cmd": args.lambda_cmd,
                "lambda_tgt": args.lambda_tgt,
            }

    ckpt_path = os.path.join(args.data_dir, "cmd_mlp_ckpt.pt")
    torch.save(best_state, ckpt_path)
    print(f"[info] Saved best CMD-MLP ckpt to {ckpt_path}, best target RMSE={best_rmse_tgt:.4f} ps")


if __name__ == "__main__":
    main()
