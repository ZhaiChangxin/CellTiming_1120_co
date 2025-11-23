# train_mlp.py  —— 方案A：SRC预训练 + TGT微调（带 cell_type one-hot）
import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mlp_model import MLPRegressor


# ===================== 基础数值特征（不含 cell_type one-hot） =====================
BASE_NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv", "stack_pu", "stack_pd",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit",   # 由 pol 构造（rise=1, fall=0）
]

TARGET_COL = "delay"


# =========================================================
# 特征处理
# =========================================================

def ensure_feature_cols(df: pd.DataFrame, numeric_cols):
    """
    - 由 pol 构造 pol_bit
    - 由 cell_type 构造 one-hot 列（列名形如 cell_type_INVX1）
    - 不存在的列补 0，NaN 也补 0
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
    df[TARGET_COL] = df[TARGET_COL].astype(np.float32)
    return df


class TimingDataset(Dataset):
    """
    简单的 csv 数据集：
      - x: numeric_cols
      - y: delay
      - 内部做标准化 (x_mean/std, y_mean/std)
    """
    def __init__(self, df: pd.DataFrame,
                 numeric_cols,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: float, y_std: float):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.numeric_cols = numeric_cols

        self.x = self.df[self.numeric_cols].values.astype(np.float32)
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


# =========================================================
# 训练 / 评估函数
# =========================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = mse(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_dataset(model, df: pd.DataFrame,
                 numeric_cols,
                 x_mean, x_std, y_mean, y_std, device, name=""):
    """
    对一个 DataFrame 做评估，返回 MAE / RMSE（单位 ps）
    """
    model.eval()
    xs = df[numeric_cols].values.astype(np.float32)
    ys = df[TARGET_COL].values.astype(np.float32)

    xs = (xs - x_mean) / x_std
    xs_t = torch.from_numpy(xs).to(device)
    ys_t = torch.from_numpy(ys).to(device)

    preds_norm = model(xs_t)
    preds = preds_norm * y_std + y_mean  # 反标准化

    diff = preds - ys_t
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    print(f"[Eval {name}] MAE={mae:.4f} ps  RMSE={rmse:.4f} ps  (N={len(df)})")
    return mae, rmse


# =========================================================
# 主流程（支持 joint / pretrain_src / finetune_tgt）
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="包含 src_delay.csv / tgt_delay_labeled.csv 的目录")
    ap.add_argument("--mode", choices=["joint", "pretrain_src", "finetune_tgt"],
                    default="joint",
                    help="joint: src+tgt_l 一起训练；"
                         "pretrain_src: 只用源域预训练；"
                         "finetune_tgt: 在目标域微调，需要 --src_ckpt")
    ap.add_argument("--src_ckpt", default=None,
                    help="finetune_tgt 模式下要加载的源域 ckpt 路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    # ---------- 1) 读数据 ----------
    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    tgt_l_csv = os.path.join(args.data_dir, "tgt_delay_labeled.csv")

    df_src = pd.read_csv(src_csv)
    df_tgt_l = pd.read_csv(tgt_l_csv)

    # =====================================================
    # 2) 准备特征列 & 归一化参数
    # =====================================================
    if args.mode != "finetune_tgt":
        # 预训练 / 联合训练：根据 src+tgt_l 的 cell_type 构造 one-hot
        df_all = pd.concat([df_src, df_tgt_l], axis=0)
        cell_types = sorted(df_all["cell_type"].astype(str).unique().tolist())
        cell_type_cols = [f"cell_type_{ct}" for ct in cell_types]

        numeric_cols = BASE_NUMERIC_COLS + cell_type_cols

        # 对三个 DataFrame 都补齐特征列
        df_src = ensure_feature_cols(df_src, numeric_cols)
        df_tgt_l = ensure_feature_cols(df_tgt_l, numeric_cols)
        df_all_feat = ensure_feature_cols(df_all, numeric_cols)

        if args.mode == "pretrain_src":
            df_train = df_src
            ckpt_name = "mlp_src_pretrain.pt"
            print("[info] mode=pretrain_src, training on SRC only")
        elif args.mode == "joint":
            df_train = df_all_feat.reset_index(drop=True)
            ckpt_name = "mlp_ckpt.pt"
            print("[info] mode=joint, training on SRC + TGT_L")
        else:
            raise ValueError("unknown mode")

        # ⚠️ 归一化参数用 SRC+TGT_L 的联合数据（加好特征之后的 df_all_feat）
        x_all = df_all_feat[numeric_cols].astype(np.float32)
        y_all = df_all_feat[[TARGET_COL]].astype(np.float32)

        x_mean = x_all.mean(0).values
        x_std = (x_all.std(0).values + 1e-9)
        y_mean = float(y_all.mean().values[0])
        y_std = float(y_all.std().values[0] + 1e-9)

    else:
        # finetune_tgt：从 src_ckpt 里加载 numeric_cols 和 scaler，不重新算
        if args.src_ckpt is None:
            raise SystemExit("[error] finetune_tgt 模式需要指定 --src_ckpt")

        print(f"[info] finetune_tgt: load src ckpt from {args.src_ckpt}")
        state_src = torch.load(args.src_ckpt, map_location=device)

        numeric_cols = state_src["numeric_cols"]
        x_mean = np.array(state_src["x_mean"], dtype=np.float32)
        x_std = np.array(state_src["x_std"], dtype=np.float32)
        y_mean = float(state_src["y_mean"])
        y_std = float(state_src["y_std"])

        # 用同样的特征列和 scaler 处理数据
        df_src = ensure_feature_cols(df_src, numeric_cols)
        df_tgt_l = ensure_feature_cols(df_tgt_l, numeric_cols)

        df_train = df_tgt_l
        ckpt_name = "mlp_tgt_finetune.pt"
        print("[info] mode=finetune_tgt, training on TGT_L only (with SRC init)")

    # 保存 scaler 信息（可选）
    scaler_path = os.path.join(args.data_dir, "mlp_scaler.json")
    json.dump(
        {
            "numeric_cols": numeric_cols,
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
        },
        open(scaler_path, "w"),
        indent=2
    )
    print(f"[info] saved scaler to {scaler_path}")

    # ---------- 3) Dataset / Dataloader ----------
    train_ds = TimingDataset(df_train, numeric_cols, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
    )

    # ---------- 4) 建模型 ----------
    if args.mode == "finetune_tgt":
        # 从 src 预训练 ckpt 里恢复网络结构和权重
        hid_dim = int(state_src["hid_dim"])
        depth = int(state_src["depth"])
        model = MLPRegressor(
            in_dim=len(numeric_cols),
            hid_dim=hid_dim,
            depth=depth,
            dropout=0.0,
        ).to(device)
        model.load_state_dict(state_src["model"])
        print(f"[info] loaded src-pretrained model: hid={hid_dim}, depth={depth}")
    else:
        # 重新初始化一个新模型
        model = MLPRegressor(
            in_dim=len(numeric_cols),
            hid_dim=args.hid,
            depth=args.depth,
            dropout=0.0,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- 5) 训练 ----------
    best_rmse = 1e9
    best_state = None

    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {ep}] train MSE (norm space) = {train_loss:.6f}")

        # 附带在 SRC / TGT_L 上评估一下，方便观察迁移效果
        mae_src, rmse_src = eval_dataset(model, df_src, numeric_cols, x_mean, x_std, y_mean, y_std, device, name="SRC")
        mae_tgt, rmse_tgt = eval_dataset(model, df_tgt_l, numeric_cols, x_mean, x_std, y_mean, y_std, device, name="TGT_L")

        # 以 TGT_L 的 RMSE 作为 early stopping 指标
        if rmse_tgt < best_rmse:
            best_rmse = rmse_tgt
            best_state = {
                "model": model.state_dict(),
                "numeric_cols": numeric_cols,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean,
                "y_std": y_std,
                "hid_dim": (int(state_src["hid_dim"]) if args.mode == "finetune_tgt" else args.hid),
                "depth": (int(state_src["depth"]) if args.mode == "finetune_tgt" else args.depth),
            }

    # ---------- 6) 保存最优模型 ----------
    ckpt_path = os.path.join(args.data_dir, ckpt_name)
    torch.save(best_state, ckpt_path)
    print(f"[info] Saved best MLP ckpt to {ckpt_path}, best target RMSE={best_rmse:.4f} ps")


if __name__ == "__main__":
    main()
