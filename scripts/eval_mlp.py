import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score

# 确保同目录下有 model.py
from model import DisentangledRegressor

# ====== 配置 (与 Train 一致) ======
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


# ====== 简化版 Dataset ======
class EvalDataset(Dataset):
    def __init__(self, csv_path, vocab, x_mean, x_std, y_mean, y_std):
        df = pd.read_csv(csv_path)

        # 补全
        if "pol_bit" not in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32) if "pol" in df.columns else 0.0
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0

        # 归一化
        self.x = (df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values - x_mean) / x_std

        # ID 转换
        self.cids = []
        for name in df["cell_type"].values:
            # 评估时，如果在训练集的 vocab 里找不到，就使用默认值(如0)
            self.cids.append(vocab.get(name, 0))
        self.cids = np.array(self.cids, dtype=np.int64)

        if TARGET_COL in df.columns:
            self.y = df[TARGET_COL].astype(np.float32).values
        else:
            self.y = None

        # 保存原始值用于反归一化输出
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        y = torch.tensor(self.y[i]) if self.y is not None else None
        return torch.from_numpy(self.x[i]), y, torch.tensor(self.cids[i])


def run_eval(tag, csv_path, model, emb_layer, vocab_map, device, x_stats, y_stats):
    print(f"[Info] Evaluating {tag} on {csv_path}")
    if not os.path.exists(csv_path):
        print("  -> File not found, skipping.")
        return

    # === 修复点开始 ===
    # 将原来的单行解包拆分为两行
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats
    # === 修复点结束 ===

    ds = EvalDataset(csv_path, vocab_map, x_mean, x_std, y_mean, y_std)
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    preds = []
    gts = []

    with torch.no_grad():
        for xb, yb, cids in dl:
            xb = xb.to(device)
            cids = cids.to(device)

            # Lookup z
            zb = emb_layer(cids)

            mu, _, _, _ = model(xb, zb)

            # Tanh clamp consistent with training
            max_abs = 10.0
            mu = max_abs * torch.tanh(mu / max_abs)

            # Unscale
            mu_ps = mu.cpu().numpy() * y_std + y_mean
            preds.append(mu_ps)

            if yb is not None:
                gts.append(yb.numpy())

    y_pred = np.concatenate(preds, axis=0)

    if len(gts) > 0:
        y_true = np.concatenate(gts, axis=0)
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        r2 = r2_score(y_true, y_pred)

        print("=" * 40)
        print(f"{tag} Result:")
        print(f"  MAE : {mae:.5f}")
        print(f"  MSE : {mse:.5f}")
        print(f"  R2  : {r2:.4f}")
        print("=" * 40)

        out_df = pd.DataFrame({"pred": y_pred.flatten(), "label": y_true.flatten()})
    else:
        out_df = pd.DataFrame({"pred": y_pred.flatten()})

    out_path = os.path.join(os.path.dirname(csv_path), f"eval_result_mlp_{tag.lower()}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  -> Saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1. Load Scalers
    ss_path = os.path.join(args.data_dir, "scaler_stats.json")
    ys_path = os.path.join(args.data_dir, "y_scaler.json")

    if not os.path.exists(ss_path) or not os.path.exists(ys_path):
        raise FileNotFoundError("scaler_stats.json or y_scaler.json not found in data_dir")

    stats = json.load(open(ss_path))
    yinfo = json.load(open(ys_path))
    x_mean = np.array([stats["mean"].get(c, 0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1) for c in NUMERIC_COLS], dtype=np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])

    # 2. Load Checkpoint
    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(args.ckpt, map_location=device)

    vocab_map = state["vocab"]  # dict: name->id
    design_dim = state["design_dim"]

    # 3. Restore Model
    emb_layer = nn.Embedding(len(vocab_map), design_dim).to(device)
    emb_layer.load_state_dict(state["emb"])
    emb_layer.eval()

    # Infer hidden dim from model weight shape
    hid_dim = state["model"]["enc.0.weight"].shape[0]

    model = DisentangledRegressor(len(NUMERIC_COLS), hid=hid_dim, design_dim_override=design_dim).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # 4. Evaluate Standard Sets
    run_eval("Source", os.path.join(args.data_dir, "src_delay.csv"),
             model, emb_layer, vocab_map, device, (x_mean, x_std), (y_mean, y_std))

    run_eval("Target_Train", os.path.join(args.data_dir, "tgt_train.csv"),
             model, emb_layer, vocab_map, device, (x_mean, x_std), (y_mean, y_std))

    run_eval("Target_Val", os.path.join(args.data_dir, "tgt_val.csv"),
             model, emb_layer, vocab_map, device, (x_mean, x_std), (y_mean, y_std))

    run_eval("Target_Test", os.path.join(args.data_dir, "tgt_test.csv"),
             model, emb_layer, vocab_map, device, (x_mean, x_std), (y_mean, y_std))


if __name__ == "__main__":
    main()
