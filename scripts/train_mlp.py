# === Python代码文件: train_mlp.py ===
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from itertools import cycle

from model import DisentangledRegressor, convert_to_bayesian

NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp", "wp_over_wn", "wp_sum", "wn_sum", "is_inv",
    "stack_pu", "stack_pd", "log_slew", "log_cap", "req_p", "req_n", "rc_p", "rc_n",
    "rc_eff", "req_eff", "inv_v", "inv_temp", "pn_balance", "pol_bit",
]
TARGET_COL = "delay"


# 【关键修正】: 鲁棒的 NLL 计算函数
def robust_nll(y, mu, log_var):
    # 1. 截断 log_var，防止 sigma^2 过小导致除以零，或过大导致溢出
    # range: [-5, 5] -> sigma: [0.006, 148]
    log_var = torch.clamp(log_var, min=-5.0, max=5.0)

    sigma2 = torch.exp(log_var)
    loss = 0.5 * (log_var + (y - mu) ** 2 / sigma2)
    return loss.mean()


class TabularDataset(Dataset):
    def __init__(self, csv_path, x_mean, x_std, y_mean, y_std):
        if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)

        if "pol_bit" not in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32) if "pol" in df.columns else 0.0
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0

        x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        # 加上 1e-6 防止标准差为0
        self.x = (x_raw - x_mean) / (x_std + 1e-6)

        y_raw = df[TARGET_COL].astype(np.float32).values
        self.y = (y_raw - y_mean) / (y_std + 1e-6)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


def load_scalers(data_dir):
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")
    if not os.path.exists(ss_path): return 0.0, 1.0, 0.0, 1.0
    stats = json.load(open(ss_path, "r"))
    yinfo = json.load(open(ys_path, "r"))
    x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    return x_mean, x_std, y_mean, y_std


# ==========================================
#      STAGE 1: Source Pretraining
# ==========================================
def run_stage1(args, model, device, train_loader, save_path):
    print(f"\n[Stage 1] Pretraining on Source... (Epochs: {args.s1_epochs})")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.s1_epochs):
        epoch_loss = 0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            mu, logv, _, _ = model(xb)

            # Stage 1: 使用 MSE 辅助训练，比 NLL 更稳定，先让均值收敛
            # 或者使用 robust_nll
            loss = robust_nll(yb, mu, logv)

            loss.backward()
            # 【关键修正】: 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item() * len(yb)
            count += len(yb)

        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch + 1}/{args.s1_epochs} | Loss: {epoch_loss / count:.4f}")

    torch.save({"model": model.state_dict()}, save_path)
    print("  -> Stage 1 Checkpoint saved.")


# ==========================================
#      STAGE 3: Target Fine-tuning
# ==========================================
def run_stage3(args, model, device, tgt_loader, src_loader, save_path):
    print(f"\n[Stage 3] Target Fine-tuning... (Epochs: {args.s3_epochs})")

    # 1. 转换模型
    model = convert_to_bayesian(model, prior_sigma=0.1).to(device)

    # 2. 设置优化器，降低学习率
    params = [
        {"params": model.enc.parameters(), "lr": args.lr * 0.01},
        {"params": model.mu.parameters(), "lr": args.lr * 0.05},
        {"params": model.log_var.parameters(), "lr": args.lr * 0.05}
    ]
    optimizer = optim.Adam(params)
    model.train()

    src_iter = cycle(src_loader)

    for epoch in range(args.s3_epochs):
        epoch_nll = 0
        epoch_kl = 0
        count = 0
        # KL 退火：从 0 开始缓慢增加
        beta = min(1.0, epoch / max(1, args.s3_epochs // 2)) * 0.01

        for xb_tgt, yb_tgt in tgt_loader:
            try:
                xb_src, yb_src = next(src_iter)
            except StopIteration:
                src_iter = cycle(src_loader)
                xb_src, yb_src = next(src_iter)

            xb_tgt, yb_tgt = xb_tgt.to(device), yb_tgt.to(device)
            xb_src, yb_src = xb_src.to(device), yb_src.to(device)

            optimizer.zero_grad()

            # Target Forward
            mu_t, logv_t, _, _ = model(xb_tgt)
            nll_tgt = robust_nll(yb_tgt, mu_t, logv_t)

            # Source Replay (Auxiliary)
            mu_s, logv_s, _, _ = model(xb_src)
            nll_src = robust_nll(yb_src, mu_s, logv_s)

            kl_loss = model.get_bayesian_kl() / (len(tgt_loader.dataset) + len(src_loader.dataset))

            # Total Loss
            loss = nll_tgt + 0.1 * nll_src + beta * kl_loss

            # 检查 loss 是否为 nan，如果是，跳过这一步（最后一道防线）
            if torch.isnan(loss):
                print(f"[Warning] NaN loss detected at epoch {epoch}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()

            # 【关键修正】: 梯度裁剪，防止权重爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_nll += nll_tgt.item() * len(yb_tgt)
            epoch_kl += kl_loss.item() * len(yb_tgt)
            count += len(yb_tgt)

        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch + 1}/{args.s3_epochs} | NLL(T): {epoch_nll / count:.4f} | KL: {epoch_kl / count:.4f}")

    torch.save({"model": model.state_dict()}, save_path)
    print("  -> Stage 3 Checkpoint saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./output_physics_fix")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--s1_epochs", type=int, default=50)
    parser.add_argument("--s3_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    print("[Info] Loading Scalers...")
    x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)

    src_ds = TabularDataset(os.path.join(args.data_dir, "src_delay.csv"), x_mean, x_std, y_mean, y_std)
    tgt_ds = TabularDataset(os.path.join(args.data_dir, "tgt_train.csv"), x_mean, x_std, y_mean, y_std)

    src_loader = DataLoader(src_ds, batch_size=128, shuffle=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=32, shuffle=True)

    model = DisentangledRegressor(len(NUMERIC_COLS), args.hid).to(device)

    ckpt_s1 = os.path.join(args.save_dir, "ckpt_s1.pt")
    ckpt_s3 = os.path.join(args.save_dir, "ckpt_s3.pt")

    run_stage1(args, model, device, src_loader, ckpt_s1)

    print("\n[Loading Stage 1 Checkpoint for Stage 3...]")
    try:
        s1 = torch.load(ckpt_s1)
    except:
        s1 = torch.load(ckpt_s1, weights_only=False)

    model.load_state_dict(s1["model"])
    run_stage3(args, model, device, tgt_loader, src_loader, ckpt_s3)

    print("\n[Done] All stages finished.")


if __name__ == "__main__":
    main()
