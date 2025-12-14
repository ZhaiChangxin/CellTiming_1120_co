import os
import json
import argparse
import time
import copy
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# === 项目依赖 ===
# 请确保同级目录下有: model.py, hgat.py, spi2graph.py, losses.py
from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
from losses import total_loss

# ====== 全局配置 ======
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
#      工具函数 & Dataset 定义
# ==========================================

def load_scalers(data_dir):
    """加载归一化统计数据"""
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")

    if not os.path.exists(ss_path) or not os.path.exists(ys_path):
        raise FileNotFoundError("找不到 scaler_stats.json 或 y_scaler.json，请先运行数据预处理。")

    stats = json.load(open(ss_path, "r"))
    yinfo = json.load(open(ys_path, "r"))

    x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])

    return x_mean, x_std, y_mean, y_std


def _infer_hgat_hid(sd: dict) -> int:
    """从 state_dict 推断 hidden dim"""
    if sd is None: return 64
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    return 64


# --- Stage 1 Dataset (源域: 实时构图或加载图) ---
class Stage1Dataset(Dataset):
    def __init__(self, csv_path, data_dir, x_mean, x_std, y_mean, y_std):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir

        # 补全 pol_bit
        if "pol_bit" not in self.df.columns:
            self.df["pol_bit"] = (self.df["pol"].astype(str) == "rise").astype(
                np.float32) if "pol" in self.df.columns else 0.0
        for c in NUMERIC_COLS:
            if c not in self.df.columns: self.df[c] = 0.0

        x_raw = self.df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        self.x = (x_raw - x_mean) / x_std

        y_raw = self.df[TARGET_COL].astype(np.float32).values
        self.y = (y_raw - y_mean) / y_std

        self.cell_types = self.df["cell_type"].values

        # 加载 source spice map
        with open(os.path.join(data_dir, "meta.json"), "r") as f:
            self.src_map = json.load(f).get("src_spi_by_cell", {})

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i]), self.cell_types[i]


# --- Stage 2 Dataset (目标域: 纯向量) ---
class Stage2Dataset(Dataset):
    def __init__(self, csv_path, x_mean, x_std, y_mean, y_std):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)

        if "pol_bit" not in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32) if "pol" in df.columns else 0.0
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0

        x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        self.x = (x_raw - x_mean) / x_std
        y_raw = df[TARGET_COL].astype(np.float32).values
        self.y = (y_raw - y_mean) / y_std
        self.cell_types = df["cell_type"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i]), self.cell_types[i]


# --- 辅助：预计算 Embedding ---
def precompute_z(data_dir, spice_file, mapping_key, enc, device):
    """
    通用函数：根据 meta.json 中的 mapping_key (如 tgt_subckt_by_cell)
    解析 spice_file，利用 enc 计算出 embedding 字典。
    """
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    mapping = meta.get(mapping_key, {})
    if not mapping:
        print(f"[Warn] No mapping found for key '{mapping_key}' in meta.json")
        return {}

    # 确定 SPICE 路径
    if not os.path.exists(spice_file):
        # 尝试拼接 data_dir
        cand = os.path.join(data_dir, spice_file)
        if os.path.exists(cand):
            spice_file = cand
        else:
            print(f"[Warn] SPICE file not found: {spice_file}")
            return {}

    print(f"[Info] Parsing SPICE: {spice_file}")
    sp_text = open(spice_file, "r", encoding="utf-8", errors="ignore").read()

    # 正则提取 subckt
    def get_subckt(name):
        patt = re.compile(r"\s*\.subckt\s+%s\b(.*?)\.ends\b" % re.escape(name), re.DOTALL | re.IGNORECASE)
        m = patt.search(sp_text)
        return m.group(0) if m else ""

    z_dict = {}
    enc.eval()
    cnt = 0
    with torch.no_grad():
        for ctype, sub_name in mapping.items():
            txt = get_subckt(sub_name)

            if not txt: continue

            devs = parse_transistors_spice(txt)
            _, pins = parse_top_subckt_pins(txt)
            if not devs: continue

            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}

            z = enc(g, feats)
            if z.dim() == 1: z = z.unsqueeze(0)
            z_dict[ctype] = z
            cnt += 1

    print(f"[Info] Pre-computed Z for {cnt} cells.")
    return z_dict


# ==========================================
#      Stage 1: 源域预训练 (Source Pretrain)
# ==========================================
def run_stage1_pretraining(args, device, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 50)
    print(" >>> STAGE 1: Source Domain Pre-training <<<")
    print("=" * 50)

    # 1. 准备数据
    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Stage 1 需要源域数据: {src_csv}")

    ds = Stage1Dataset(src_csv, args.data_dir, x_mean, x_std, y_mean, y_std)
    loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=lambda x: x)

    # 2. 初始化模型
    in_map = {"NET": 4, "PMOS": 2, "NMOS": 2}
    enc = HGATDesignEncoder(in_dim_map=in_map, hid=args.hid, out=args.design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=args.design_dim).to(
        device)

    # 3. 优化器 (训练所有参数)
    optimizer = optim.Adam(list(enc.parameters()) + list(model.parameters()), lr=args.lr)

    scheduler = None
    if getattr(args, "auto_lr", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
            min_lr=args.min_lr
        )


    # 4. 预计算源域 Z
    print("[Stage 1] Caching Source Graphs (CPU Pre-processing)...")
    graph_cache = {}  # cell_type -> (g, feats)

    with open(os.path.join(args.data_dir, "meta.json"), 'r') as f:
        src_map = json.load(f).get("src_spi_by_cell", {})

    total_cells = len(src_map)
    start_time = time.time()

    for idx, (ct, path) in enumerate(src_map.items()):
        if idx % 10 == 0 or idx == total_cells - 1:
            elapsed = time.time() - start_time
            print(f"  -> Parsing {idx + 1}/{total_cells} cells... ({elapsed:.1f}s)", end='\r')

        full_path = path if os.path.exists(path) else os.path.join(args.data_dir, path)
        if os.path.exists(full_path):
            txt = open(full_path, 'r').read()
            devs = parse_transistors_spice(txt)
            _, pins = parse_top_subckt_pins(txt)
            if devs:
                g, feats, _ = build_dgl_graph_from_devs(devs, pins)
                graph_cache[ct] = (g.to(device), {k: v.to(device) for k, v in feats.items()})

    print(f"\n[Stage 1] Cached {len(graph_cache)} graphs successfully.")

    # 5. 训练循环
    enc.train()
    model.train()

    best_loss = float('inf')
    bad_epochs = 0

    for epoch in range(args.s1_epochs):
        epoch_loss_val = 0  # 避免变量名冲突
        count = 0

        for batch in loader:
            xs, ys, cts = zip(*batch)
            xs = torch.stack(xs).to(device)
            ys = torch.stack(ys).to(device)

            z_list = []
            valid_indices = []

            for i, ct in enumerate(cts):
                if ct in graph_cache:
                    g, feats = graph_cache[ct]
                    z = enc(g, feats)
                    if z.dim() == 1: z = z.unsqueeze(0)
                    z_list.append(z)
                    valid_indices.append(i)

            if not z_list: continue

            zb = torch.cat(z_list, dim=0)
            xb = xs[valid_indices]
            yb = ys[valid_indices]

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=0.05)

            loss.backward()
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(model.parameters()), args.grad_clip)
            optimizer.step()

            epoch_loss_val += loss.item() * len(yb)
            count += len(yb)

        avg_loss = epoch_loss_val / (count + 1e-6)
        if scheduler is not None:
            scheduler.step(avg_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0:
            print(f"  [S1] Epoch {epoch + 1}/{args.s1_epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")
        else:
            # keep old print cadence quiet
            pass

        # best ckpt + early stop

            print(f"  [S1] Epoch {epoch + 1}/{args.s1_epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
            save_path = os.path.join(args.save_dir, "ckpt_stage1_best.pt")
            torch.save({
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_map,
                "design_dim": args.design_dim
            }, save_path)
        else:
            bad_epochs += 1
            if getattr(args, "early_patience", 0) and args.early_patience > 0 and bad_epochs >= args.early_patience:
                print(f"  [S1] Early stop at epoch {epoch+1} (no improvement for {bad_epochs} epochs).")
                break

    print(f"[Stage 1] Finished. Checkpoint saved to: {os.path.join(args.save_dir, 'ckpt_stage1_best.pt')}")
    return os.path.join(args.save_dir, "ckpt_stage1_best.pt")


# ==========================================
#      Stage 2: 目标域微调 (Target Transfer)
# ==========================================
def run_stage2_transfer(args, device, src_ckpt_path, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 50)
    print(" >>> STAGE 2: Target Domain Fine-tuning (Transfer) <<<")
    print("=" * 50)

    if not os.path.exists(src_ckpt_path):
        raise FileNotFoundError(f"Stage 2 找不到源域模型: {src_ckpt_path}")

    # 1. 加载 Checkpoint
    print(f"[S2] Loading Source Checkpoint: {src_ckpt_path}")
    state = torch.load(src_ckpt_path, map_location=device)

    # 2. 恢复模型结构
    design_dim = int(state.get("design_dim", 64))
    enc_hid = _infer_hgat_hid(state["enc"])
    in_map = state.get("hgat_in_dim_map", {"NET": 4, "PMOS": 2, "NMOS": 2})

    enc = HGATDesignEncoder(in_dim_map=in_map, hid=enc_hid, out=design_dim).to(device)
    enc.load_state_dict(state["enc"])

    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=design_dim).to(device)

    try:
        model.load_state_dict(state["model"], strict=True)
    except:
        print("[Warn] MLP 结构不完全匹配，尝试非严格加载或调整 shape...")
        model.load_state_dict(state["model"], strict=False)

    model = model.to(device)

    # 3. 关键：冻结 GNN，只训练 MLP
    print("[S2] Freezing HGAT Encoder...")
    for p in enc.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.5)

    scheduler = None
    if getattr(args, "auto_lr", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
            min_lr=args.min_lr
        )

    # 4. 预计算目标域 Z
    z_map = precompute_z(args.data_dir, args.tgt_spice, "tgt_subckt_by_cell", enc, device)

    # 5. 加载数据
    train_csv = os.path.join(args.data_dir, "tgt_train.csv")
    val_csv = os.path.join(args.data_dir, "tgt_val.csv")

    if not os.path.exists(train_csv):
        raise RuntimeError("Missing tgt_train.csv")

    train_ds = Stage2Dataset(train_csv, x_mean, x_std, y_mean, y_std)
    val_ds = Stage2Dataset(val_csv, x_mean, x_std, y_mean, y_std) if os.path.exists(val_csv) else None

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return torch.stack(xs), torch.stack(ys), cts

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=my_collate) if val_ds else None

    print(f"[S2] Train Samples: {len(train_ds)}, Val Samples: {len(val_ds) if val_ds else 0}")

    # 6. 微调循环
    best_mae = float('inf')
    bad_epochs = 0

    for epoch in range(args.s2_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0  # <--- 修正点：改名了
        for xb, yb, cts in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            z_list = [z_map[ct] if ct in z_map else torch.zeros(1, design_dim, device=device) for ct in cts]
            zb = torch.cat(z_list, dim=0)

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            # 现在可以正常调用函数 total_loss 了
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=0.01)

            loss.backward()
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(model.parameters()), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        avg_train_loss = epoch_loss / len(train_ds)

        # --- Val ---
        val_mae = 0.0
        if val_dl:
            model.eval()
            err_sum = 0
            with torch.no_grad():
                for xb, yb, cts in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    z_list = [z_map[ct] if ct in z_map else torch.zeros(1, design_dim, device=device) for ct in cts]
                    zb = torch.cat(z_list, dim=0)

                    mu, _, _, _ = model(xb, zb)

                    pred_ps = mu * y_std + y_mean
                    true_ps = yb * y_std + y_mean

                    max_abs = 10.0
                    mu_t = max_abs * torch.tanh(mu / max_abs)
                    pred_ps_t = mu_t * y_std + y_mean

                    err_sum += torch.abs(pred_ps_t - true_ps).sum().item()
            val_mae = err_sum / len(val_ds)

        metric = val_mae if val_dl else avg_train_loss
        if scheduler is not None:
            scheduler.step(metric)
        cur_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0:
            print(
                f"  [S2] Epoch {epoch + 1}/{args.s2_epochs} | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} ps | LR: {cur_lr:.2e}")

        if val_mae < best_mae:
            best_mae = val_mae
            bad_epochs = 0
            save_path = os.path.join(args.save_dir, "ckpt_transfer_best.pt")
            torch.save({
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_map,
                "design_dim": design_dim
            }, save_path)
        else:
            bad_epochs += 1
            if getattr(args, "early_patience", 0) and args.early_patience > 0 and bad_epochs >= args.early_patience:
                print(f"  [S2] Early stop at epoch {epoch+1} (no improvement for {bad_epochs} epochs).")
                break

    print(f"[Stage 2] Finished. Best Val MAE: {best_mae:.4f} ps")
    print(f"Final Model saved to: {os.path.join(args.save_dir, 'ckpt_transfer_best.pt')}")


# ==========================================
#      Main Entry
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="数据根目录")
    parser.add_argument("--save_dir", default="./output", help="模型保存目录")
    parser.add_argument("--tgt_spice", default="asap7.sp", help="目标域 SPICE 文件名 (相对于 data_dir)")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "transfer", "all"],
                        help="运行模式: pretrain(只S1), transfer(只S2), all(S1+S2)")
    parser.add_argument("--src_ckpt", type=str, default="",
                        help="指定用于迁移的源域模型路径 (mode=transfer时必填，mode=all时自动生成)")
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--design_dim", type=int, default=64)
    parser.add_argument("--s1_epochs", type=int, default=50, help="Stage 1 Epochs")
    parser.add_argument("--s2_epochs", type=int, default=200, help="Stage 2 Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="base learning rate")
    # --- Auto LR / Early stop ---
    parser.add_argument("--auto_lr", action="store_true", help="enable ReduceLROnPlateau scheduler")
    parser.add_argument("--lr_patience", type=int, default=10, help="epochs w/o improvement before reducing LR")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="LR *= factor on plateau")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum LR for scheduler")
    parser.add_argument("--early_patience", type=int, default=30, help="early stop after N bad epochs (0=disable)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="clip grad norm (0=disable)")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("[Info] Loading Scalers...")
    x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)

    current_ckpt = args.src_ckpt

    if args.mode in ["pretrain", "all"]:
        current_ckpt = run_stage1_pretraining(args, device, x_mean, x_std, y_mean, y_std)

    if args.mode in ["transfer", "all"]:
        if not current_ckpt or not os.path.exists(current_ckpt):
            print("[Error] 无法运行 Stage 2，因为没有有效的源域 Checkpoint。")
            if args.mode == "transfer":
                print("请使用 --src_ckpt 指定预训练模型路径。")
            return

        run_stage2_transfer(args, device, current_ckpt, x_mean, x_std, y_mean, y_std)


if __name__ == "__main__":
    main()
