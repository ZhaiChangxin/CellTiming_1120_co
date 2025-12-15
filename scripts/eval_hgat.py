# === Python代码文件: eval_hgat.py ===
import os
import json
import argparse
import re
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 引入项目依赖
from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins

# ====== 配置 ======
NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp", "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv", "stack_pu", "stack_pd", "log_slew", "log_cap",
    "req_p", "req_n", "rc_p", "rc_n", "rc_eff", "req_eff",
    "inv_v", "inv_temp", "pn_balance", "pol_bit",
]
TARGET_COL = "delay"


# ====== 辅助函数 ======
def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    lines = sp_text.splitlines(keepends=True)
    collecting = False
    buf = []
    patt_begin = re.compile(r"\s*\.subckt\s+%s\b" % re.escape(subckt_name), re.IGNORECASE)
    patt_end = re.compile(r"\s*\.ends\b", re.IGNORECASE)
    for line in lines:
        if not collecting:
            if patt_begin.search(line):
                collecting = True
                buf.append(line)
        else:
            buf.append(line)
            if patt_end.match(line):
                break
    return "".join(buf) if buf else ""


class EvalDataset(Dataset):
    def __init__(self, csv_path: str, x_mean, x_std):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)

        # 补全列
        if "pol_bit" not in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32) if "pol" in df.columns else 0.0
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0

        # X 进行归一化 (Model 输入需要)
        x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        self.x = (x_raw - x_mean) / x_std

        # Y 保持原始值或读取后在外部处理？
        # 这里为了 Dataset 统一，我们先读取 normalized Y (假设 CSV 是原始值，需要手动归一化用于 Loss，但评估时我们需要原始值)
        # 策略：Dataset 返回 Normalized X 和 Raw Y，我们在 eval loop 里反归一化 Pred，然后和 Raw Y 比较
        if TARGET_COL in df.columns:
            self.y_raw = df[TARGET_COL].astype(np.float32).values
        else:
            self.y_raw = None

        self.cell_types = df["cell_type"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), self.y_raw[i] if self.y_raw is not None else 0.0, self.cell_types[i]


def eval_collate(batch):
    xs, ys_raw, cts = zip(*batch)
    return torch.stack(xs), np.array(ys_raw), cts


# ====== Embedding 计算 ======
def prepare_embeddings(data_dir, spice_path, mapping, enc, device):
    """计算图 Embedding (Z)"""
    # ... (逻辑同前，保持不变) ...
    if not os.path.exists(spice_path) and not str(spice_path).endswith("placeholder"):
        # 这种情况下可能是 Source (分立文件)
        pass

    z_dict = {}
    enc.eval()

    # 读取 Target 大文件内容 (如果是 Target)
    sp_text = ""
    if os.path.isfile(spice_path):
        sp_text = open(spice_path, "r", encoding="utf-8", errors="ignore").read()

    with torch.no_grad():
        for cell_type, sub_name_or_path in mapping.items():
            # 区分 Source (filepath) 和 Target (subckt name)
            sub_txt = ""
            if sub_name_or_path.endswith(".sp") or sub_name_or_path.endswith(".spice"):
                # Source Mode
                full_p = sub_name_or_path if os.path.exists(sub_name_or_path) else os.path.join(data_dir,
                                                                                                sub_name_or_path)
                if os.path.exists(full_p):
                    sub_txt = open(full_p, 'r').read()
            else:
                # Target Mode
                if sp_text:
                    sub_txt = extract_subckt_text(sp_text, sub_name_or_path)

            if not sub_txt: continue

            # 解析
            devs = parse_transistors_spice(sub_txt)
            _, pins = parse_top_subckt_pins(sub_txt)
            if devs:
                # 使用更新后的 hgat 构图 (带 num_nodes_dict)
                g, feats, _ = build_dgl_graph_from_devs(devs, pins)
                z = enc(g.to(device), {k: v.to(device) for k, v in feats.items()})
                if z.dim() == 1: z = z.unsqueeze(0)
                z_dict[cell_type] = z.cpu()  # 存到 CPU 节省显存，batch 时再转 GPU

    return z_dict


# ====== 核心评估流程 ======
def run_evaluation(tag, csv_path, model, z_map, device, x_mean, x_std, y_mean, y_std, design_dim, output_dir):
    print(f"[Info] Evaluating {tag} on: {csv_path}")

    # 1. Dataset
    # 注意：Dataset 内部用 x_mean/x_std 归一化 X
    ds = EvalDataset(csv_path, x_mean.cpu().numpy(), x_std.cpu().numpy())
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=eval_collate)

    preds_real = []
    labels_real = []

    model.eval()
    with torch.no_grad():
        for xs, ys_raw, cts in loader:
            xs = xs.to(device)

            # 组装 Z
            z_batch = []
            for ct in cts:
                if ct in z_map:
                    z_batch.append(z_map[ct].to(device))
                else:
                    z_batch.append(torch.zeros(1, design_dim).to(device))
            zb = torch.cat(z_batch, dim=0)

            # Forward
            # BNN 注意: eval 模式下 BayesianLinear 使用 weight_mu 计算，是确定性的
            mu, logv, _, _ = model(xs, zb)

            # 反归一化 Prediction: Pred_real = Pred_norm * y_std + y_mean
            mu_real = mu.cpu().numpy().flatten() * y_std + y_mean

            preds_real.append(mu_real)
            labels_real.append(ys_raw)

    y_pred = np.concatenate(preds_real)
    y_true = np.concatenate(labels_real)

    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 打印日志 (匹配用户格式)
    print("=" * 40)
    print(f"{tag} Test Result ({len(ds)} samples):")
    print(f"  MAE : {mae:.5f} (ps)")
    print(f"  MSE : {mse:.5f} (ps^2)")
    print(f"  R2  : {r2:.4f}")
    print("=" * 40)

    # 保存结果
    save_name = f"eval_result_{tag.lower()}.csv"
    save_path = os.path.join(output_dir, save_name)
    res_df = pd.DataFrame({
        "pred": y_pred,
        "label": y_true,
        "diff": y_pred - y_true
    })
    res_df.to_csv(save_path, index=False)
    print(f"[Info] {tag} results saved to {save_path}\n")


# ====== Main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tgt_spice", default="asap7.sp")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. 加载统计信息 (Scalers)
    try:
        ss_path = os.path.join(args.data_dir, "scaler_stats.json")
        ys_path = os.path.join(args.data_dir, "y_scaler.json")
        stats = json.load(open(ss_path))
        yinfo = json.load(open(ys_path))

        x_mean = torch.tensor([stats["mean"].get(c, 0.) for c in NUMERIC_COLS]).float()
        x_std = torch.tensor([stats["std"].get(c, 1.) for c in NUMERIC_COLS]).float()
        y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    except Exception as e:
        print(f"[Error] Failed to load scalers: {e}")
        return

    # 2. 加载模型
    print(f"Loading checkpoint: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
    except:
        # 兼容某些环境的 pickle 问题
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # 从 ckpt 恢复配置
    conf = ckpt.get("config", {"hid": 128, "design_dim": 64})
    hid = conf["hid"]
    design_dim = conf["design_dim"]

    # 初始化网络结构
    enc = HGATDesignEncoder(in_dim_map={"NET": 4, "PMOS": 2, "NMOS": 2},
                            hid=hid, out=design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=hid, design_dim_override=design_dim).to(device)

    # 检测是否需要转为 BNN (Phase 3 产物)
    state_dict = ckpt["model"]
    is_bnn = any("weight_rho" in k for k in state_dict.keys())
    if is_bnn:
        print(" >>> Detected BNN checkpoint. Converting model structure...")
        model.convert_to_bnn()

    enc.load_state_dict(ckpt["enc"])
    model.load_state_dict(state_dict)

    # 3. 预计算 Graph Embeddings (Z)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))

    # Source Z
    print("Preparing Source Embeddings...")
    z_src = prepare_embeddings(args.data_dir, "placeholder", meta["src_spi_by_cell"], enc, device)

    # Target Z
    print("Preparing Target Embeddings...")
    tgt_spice_path = os.path.join(args.data_dir, args.tgt_spice)
    z_tgt = prepare_embeddings(args.data_dir, tgt_spice_path, meta["tgt_subckt_by_cell"], enc, device)

    # 4. 执行评估任务列表
    # 任务格式: (Tag, CSV文件名, 使用的Z映射)
    tasks = [
        ("Source", "src_delay.csv", z_src),
        ("Target_Train", "tgt_train.csv", z_tgt),
        ("Target_Val", "tgt_val.csv", z_tgt),
        ("Target_Test", "tgt_test.csv", z_tgt),
    ]

    for tag, csv_name, z_map in tasks:
        csv_full_path = os.path.join(args.data_dir, csv_name)
        if os.path.exists(csv_full_path):
            run_evaluation(
                tag=tag,
                csv_path=csv_full_path,
                model=model,
                z_map=z_map,
                device=device,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                design_dim=design_dim,
                output_dir=args.data_dir
            )
        else:
            print(f"[Warn] Skipping {tag}, file not found: {csv_full_path}")


if __name__ == "__main__":
    main()
