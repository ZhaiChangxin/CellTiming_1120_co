import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# === 复用原有依赖 ===
# 确保目录下有 model.py 和 losses.py
from model import DisentangledRegressor
from losses import total_loss

# ====== 全局配置 (保持一致) ======
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


# ==========================================
#      辅助类：词表管理 (Vocab)
# ==========================================
class CellVocab:
    def __init__(self, data_dir):
        self.stoi = {}
        self.itos = {}
        self._build(data_dir)

    def _build(self, data_dir):
        meta_path = os.path.join(data_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta.json in {data_dir}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # 收集所有可能出现的 cell_type (源域 + 目标域)
        cells = set()

        # 1. Source cells
        src_map = meta.get("src_spi_by_cell", {})
        cells.update(src_map.keys())

        # 2. Target cells
        tgt_map = meta.get("tgt_subckt_by_cell", {})
        cells.update(tgt_map.keys())

        # 排序以保证 ID 确定性
        for idx, name in enumerate(sorted(list(cells))):
            self.stoi[name] = idx
            self.itos[idx] = name

        print(f"[Vocab] Built vocabulary with {len(self.stoi)} unique cell types.")

    def __len__(self):
        return len(self.stoi)

    def get_id(self, name):
        # 如果遇到 meta.json 里没登记的 cell，返回 -1 (需要在 Dataset 里处理)
        return self.stoi.get(name, -1)


# ==========================================
#      Dataset (纯表格)
# ==========================================
class TabularDataset(Dataset):
    def __init__(self, csv_path, vocab, x_mean, x_std, y_mean, y_std):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")

        df = pd.read_csv(csv_path)

        # 预处理逻辑与 train_hgat.py 保持完全一致
        if "pol_bit" not in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32) if "pol" in df.columns else 0.0

        for c in NUMERIC_COLS:
            if c not in df.columns:
                df[c] = 0.0

        # 标准化 X
        x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        self.x = (x_raw - x_mean) / x_std

        # 标准化 Y
        y_raw = df[TARGET_COL].astype(np.float32).values
        self.y = (y_raw - y_mean) / y_std

        # 处理 Cell Type ID
        self.cell_ids = []
        unknown_count = 0
        for name in df["cell_type"].values:
            cid = vocab.get_id(name)
            if cid < 0:
                cid = 0  # Fallback: 映射到第一个 cell，或者你可以选择抛出异常
                unknown_count += 1
            self.cell_ids.append(cid)

        self.cell_ids = np.array(self.cell_ids, dtype=np.int64)

        if unknown_count > 0:
            print(f"[Warn] Found {unknown_count} unknown cell types in {os.path.basename(csv_path)}, mapped to ID 0.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i]),
            torch.tensor(self.y[i]),
            torch.tensor(self.cell_ids[i])
        )


def load_scalers(data_dir):
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")
    if not os.path.exists(ss_path):
        raise FileNotFoundError("Missing scaler_stats.json")

    stats = json.load(open(ss_path, "r"))
    yinfo = json.load(open(ys_path, "r"))

    x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    return x_mean, x_std, y_mean, y_std


# ==========================================
#      Stage 1: Pretrain (Source)
# ==========================================
def run_stage1(args, vocab, device, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 50)
    print(" >>> STAGE 1: MLP Source Pre-training <<<")
    print("=" * 50)

    # 1. Dataset
    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    ds = TabularDataset(src_csv, vocab, x_mean, x_std, y_mean, y_std)
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    # 2. Model: Embedding + Regressor
    # 使用 Embedding 替代 HGATEncoder
    emb_layer = nn.Embedding(len(vocab), args.design_dim).to(device)

    # 复用你的 DisentangledRegressor
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=args.design_dim).to(
        device)

    # 3. Optimize all
    optimizer = optim.Adam(list(emb_layer.parameters()) + list(model.parameters()), lr=args.lr)

    scheduler = None
    if args.auto_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    # 4. Loop
    emb_layer.train()
    model.train()
    best_loss = float('inf')

    for epoch in range(args.s1_epochs):
        epoch_loss = 0
        count = 0

        for xb, yb, cids in loader:
            xb, yb, cids = xb.to(device), yb.to(device), cids.to(device)

            # Look up z
            zb = emb_layer(cids)  # [B, design_dim]

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=0.05)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(yb)
            count += len(yb)

        avg_loss = epoch_loss / count
        if scheduler: scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  [S1] Epoch {epoch + 1}/{args.s1_epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "emb": emb_layer.state_dict(),
                "vocab": vocab.stoi,  # 保存词表以防 ID 错位
                "design_dim": args.design_dim
            }, os.path.join(args.save_dir, "ckpt_mlp_s1.pt"))

    return os.path.join(args.save_dir, "ckpt_mlp_s1.pt")


# ==========================================
#      Stage 2: Transfer (Target)
# ==========================================
def run_stage2(args, vocab, device, src_ckpt, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 50)
    print(" >>> STAGE 2: MLP Target Fine-tuning <<<")
    print("=" * 50)

    # 1. Load Pretrained
    ckpt = torch.load(src_ckpt, map_location=device)

    # Check vocab consistency
    saved_vocab = ckpt.get("vocab", {})
    if len(saved_vocab) != len(vocab):
        print("[Warn] Vocab size changed! Using current vocab but ID mapping might be off if meta.json changed.")

    emb_layer = nn.Embedding(len(vocab), args.design_dim).to(device)
    emb_layer.load_state_dict(ckpt["emb"],
                              strict=False)  # allow mismatch logic if strictly needed, usually should match

    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=args.design_dim).to(
        device)
    model.load_state_dict(ckpt["model"])

    # 2. Freeze Config
    # 在 MLP Baseline 中，如果不微调 Embedding，面对未见过的 Target Cell 将无能为力。
    # 策略 A: 冻结 MLP 权重，只微调 Embedding (类似 HGAT 冻结 Encoder，只调 Head，但这里反过来？)
    # 策略 B: 既然没有图结构泛化，通常做法是全量微调，或者用较小 LR 微调。
    # 为了与 HGAT 保持 "Transfer" 的概念，我们允许 Embedding 更新，同时微调 Regressor。

    # 对应 HGAT 的 "Freezing HGAT Encoder"：HGAT 冻结的是图特征提取器。
    # MLP 的 "Embedding" 相当于特征提取器。
    # 但如果冻结 Embedding，Target 独有的 Cell 将永远是随机初始化的向量。
    # 因此，MLP Baseline **必须** 训练 Embedding 层。

    optimizer = optim.Adam(list(emb_layer.parameters()) + list(model.parameters()), lr=args.lr * 0.5)

    # 3. Data
    train_ds = TabularDataset(os.path.join(args.data_dir, "tgt_train.csv"), vocab, x_mean, x_std, y_mean, y_std)
    val_csv = os.path.join(args.data_dir, "tgt_val.csv")
    val_ds = TabularDataset(val_csv, vocab, x_mean, x_std, y_mean, y_std) if os.path.exists(val_csv) else None

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False) if val_ds else None

    # 4. Loop
    best_mae = float('inf')

    for epoch in range(args.s2_epochs):
        # Train
        model.train()
        emb_layer.train()
        train_loss = 0
        for xb, yb, cids in train_dl:
            xb, yb, cids = xb.to(device), yb.to(device), cids.to(device)
            zb = emb_layer(cids)

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=0.01)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)

        avg_train = train_loss / len(train_ds)

        # Val
        val_mae = 0.0
        if val_ds:
            model.eval()
            emb_layer.eval()
            err_sum = 0
            with torch.no_grad():
                for xb, yb, cids in val_dl:
                    xb, yb, cids = xb.to(device), yb.to(device), cids.to(device)
                    zb = emb_layer(cids)
                    mu, _, _, _ = model(xb, zb)

                    # 反归一化
                    pred_ps = mu * y_std + y_mean
                    true_ps = yb * y_std + y_mean

                    # Tanh clamping (match HGAT logic)
                    mu_t = 10.0 * torch.tanh(mu / 10.0)
                    pred_ps_t = mu_t * y_std + y_mean

                    err_sum += torch.abs(pred_ps_t - true_ps).sum().item()
            val_mae = err_sum / len(val_ds)

        if (epoch + 1) % 10 == 0:
            print(f"  [S2] Epoch {epoch + 1}/{args.s2_epochs} | Train: {avg_train:.4f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                "model": model.state_dict(),
                "emb": emb_layer.state_dict(),
                "vocab": vocab.stoi,
                "design_dim": args.design_dim
            }, os.path.join(args.save_dir, "ckpt_mlp_s2.pt"))

    print(f"[Stage 2] Finished. Best MAE: {best_mae:.4f}")


# ==========================================
#      Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./output_mlp")
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--design_dim", type=int, default=64)
    parser.add_argument("--s1_epochs", type=int, default=50)
    parser.add_argument("--s2_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--auto_lr", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", default="all", choices=["pretrain", "transfer", "all"])
    parser.add_argument("--src_ckpt", default="")  # for transfer only mode

    args = parser.parse_args()
    device = torch.device(args.device)
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Load shared assets
    print("[Info] Building Vocab & Scalers...")
    vocab = CellVocab(args.data_dir)
    x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)

    curr_ckpt = args.src_ckpt

    if args.mode in ["pretrain", "all"]:
        curr_ckpt = run_stage1(args, vocab, device, x_mean, x_std, y_mean, y_std)

    if args.mode in ["transfer", "all"]:
        if not curr_ckpt:
            print("[Error] No source ckpt for transfer.")
            return
        run_stage2(args, vocab, device, curr_ckpt, x_mean, x_std, y_mean, y_std)


if __name__ == "__main__":
    main()
