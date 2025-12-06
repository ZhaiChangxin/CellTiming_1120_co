# eval_hgat.py
import os
import json
import argparse
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score

# 引入项目依赖
from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins

# ====== 数值列定义 (需与训练一致) ======
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


# ====== 辅助函数：从大文件中提取 subckt ======
def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    lines = sp_text.splitlines(keepends=True)
    collecting = False
    buf = []
    # 精确匹配 .subckt name
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


# ====== 模型调整工具 ======
def _infer_hgat_hid_from_state(sd: dict) -> int:
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    for k, v in sd.items():
        if k.startswith("readout.") and v.dim() == 2:
            return v.shape[1]
    return 64  # 默认兜底


def _reshape_main_net_to_ckpt(model, ckpt_model):
    """根据 ckpt 调整模型层宽"""
    import torch.nn as nn

    # 调整 Encoder
    if "enc.0.weight" in ckpt_model:
        model.enc[0] = nn.Linear(ckpt_model["enc.0.weight"].shape[1], ckpt_model["enc.0.weight"].shape[0])
    if "enc.2.weight" in ckpt_model:
        model.enc[2] = nn.Linear(ckpt_model["enc.2.weight"].shape[1], ckpt_model["enc.2.weight"].shape[0])

    # 调整 Split Node
    if "split_node.weight" in ckpt_model:
        model.split_node = nn.Linear(ckpt_model["split_node.weight"].shape[1], ckpt_model["split_node.weight"].shape[0])

    # 调整 Head
    if "head.0.weight" in ckpt_model:
        model.head[0] = nn.Linear(ckpt_model["head.0.weight"].shape[1], ckpt_model["head.0.weight"].shape[0])
    if "head.2.weight" in ckpt_model:
        model.head[2] = nn.Linear(ckpt_model["head.2.weight"].shape[1], ckpt_model["head.2.weight"].shape[0])

    # 调整 Mu/LogVar
    if "mu.weight" in ckpt_model:
        model.mu = nn.Linear(ckpt_model["mu.weight"].shape[1], 1)
        model.log_var = nn.Linear(ckpt_model["log_var.weight"].shape[1], 1)

    return model


# ====== 数据集 (包含 Cell Type) ======
class EvalDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"找不到数据集文件：{csv_path}")
        df = pd.read_csv(csv_path)

        # 1. 构造 pol_bit
        if "pol_bit" not in df.columns:
            if "pol" in df.columns:
                df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
            else:
                df["pol_bit"] = 0.0

        # 2. 补齐列
        for c in NUMERIC_COLS:
            if c not in df.columns:
                df[c] = 0.0

        self.x = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        self.cell_types = df["cell_type"].values  # 关键：读取 cell_type

        if TARGET_COL in df.columns:
            self.y = df[TARGET_COL].astype(np.float32).values
        else:
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i])
        ct = self.cell_types[i]
        y = torch.tensor(self.y[i]) if self.y is not None else None
        return x, y, ct


def eval_collate(batch):
    xs, ys, cts = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys) if ys[0] is not None else None
    return xs, ys, cts  # cts 是 tuple of strings


# ====== 核心：加载目标域图谱并计算 Z ======
def prepare_target_embeddings(data_dir, tgt_spice_path, enc, device):
    """
    根据 data_dir/meta.json 和 tgt_spice_path，
    为目标域的每种 cell_type 计算对应的 z 向量。
    """
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json: {meta_path}，无法建立映射关系。")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    tgt_map = meta.get("tgt_subckt_by_cell", {})
    if not tgt_map:
        raise RuntimeError("meta.json 中没有 tgt_subckt_by_cell 映射信息。")

    if not os.path.exists(tgt_spice_path):
        # 尝试从 meta 中获取路径
        cand = meta.get("tgt_sp_file", "")
        if cand and os.path.exists(os.path.join(data_dir, cand)):
            tgt_spice_path = os.path.join(data_dir, cand)
        elif cand and os.path.exists(cand):
            tgt_spice_path = cand
        else:
            raise FileNotFoundError(f"找不到目标域 SPICE 文件: {tgt_spice_path}")

    print(f"[Info] Parsing Target SPICE from: {tgt_spice_path}")
    sp_text = open(tgt_spice_path, "r", encoding="utf-8", errors="ignore").read()

    z_dict = {}
    enc.eval()

    print("[Info] Pre-computing Z for target cells:")
    with torch.no_grad():
        for cell_type, subckt_name in tgt_map.items():
            # 1. 提取特定 subckt 文本
            sub_txt = extract_subckt_text(sp_text, subckt_name)
            if not sub_txt:
                print(f"  [Warn] Subckt '{subckt_name}' (for {cell_type}) not found in SP file. Skipping.")
                continue

            # 2. 解析并构图
            devs = parse_transistors_spice(sub_txt)
            _, pins = parse_top_subckt_pins(sub_txt)
            if not devs:
                print(f"  [Warn] No devs found for {cell_type}.")
                continue

            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}

            # 3. 编码得到 z
            z = enc(g, feats)

            # 4. 维度修正 [D] -> [1, D]
            if z.dim() == 1:
                z = z.unsqueeze(0)

            z_dict[cell_type] = z
            print(f"  -> {cell_type}: subckt='{subckt_name}', z.shape={tuple(z.shape)}")

    return z_dict


# ====== 主流程 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="数据目录 (含 meta.json)")
    ap.add_argument("--ckpt", required=True, help="模型权重 .pt")
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--use_hgat", action="store_true", default=True)
    ap.add_argument("--hid", type=int, default=128, help="MLP 隐藏层维度")
    ap.add_argument("--tgt_spice", type=str, default="", help="ASAP7 SP 文件路径")
    ap.add_argument("--csv", type=str, default="", help="指定评测 CSV")

    args = ap.parse_args()
    device = torch.device(args.device)

    # 1. 确定评测文件
    if args.csv:
        csv_path = args.csv
    else:
        # 默认优先测有标签的 tgt 数据，算指标
        cand = [
            os.path.join(args.data_dir, "tgt_delay_labeled.csv"),
            os.path.join(args.data_dir, "tgt.csv"),
        ]
        csv_path = next((p for p in cand if os.path.isfile(p)), None)
        if not csv_path:
            raise FileNotFoundError("未找到默认评测 CSV，请用 --csv 指定")

    print(f"[Info] Evaluating on: {csv_path}")

    # 2. 读取 Scaler
    ss_path = os.path.join(args.data_dir, "scaler_stats.json")
    ys_path = os.path.join(args.data_dir, "y_scaler.json")
    if os.path.exists(ss_path) and os.path.exists(ys_path):
        stats = json.load(open(ss_path))
        yinfo = json.load(open(ys_path))
        x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
        x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
        y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    else:
        raise FileNotFoundError("找不到 scaler json 文件，无法反归一化。")

    # 3. 数据加载
    ds = EvalDataset(csv_path)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=eval_collate)

    # 4. 加载模型权重
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        ckpt_model = state["model"]
        ckpt_enc = state.get("enc", None)
        ckpt_inmap = state.get("hgat_in_dim_map", None)
        design_dim = int(state.get("design_dim", 64))
    else:
        raise RuntimeError("Ckpt 格式不符合预期 (缺少 model/enc 键)")

    # 5. 初始化 HGAT 并加载权重
    if ckpt_inmap is None:
        # 兜底：如果 ckpt 没存 dim_map，只能硬编码 (不推荐，但在 MVP 阶段可能发生)
        print("[Warn] ckpt 中没有 hgat_in_dim_map，使用默认值尝试...")
        ckpt_inmap = {'NET': 4, 'PMOS': 2, 'NMOS': 2}

    enc_hid = _infer_hgat_hid_from_state(ckpt_enc)
    enc = HGATDesignEncoder(in_dim_map=ckpt_inmap, hid=enc_hid, out=64).to(device)
    enc.load_state_dict(ckpt_enc, strict=True)

    # 6. 预计算 Target 域的所有 Z 向量
    # 这里需要 meta.json 来指导如何从 args.tgt_spice 中切分 cell
    z_map = prepare_target_embeddings(args.data_dir, args.tgt_spice, enc, device)

    # 7. 初始化回归模型
    model = DisentangledRegressor(
        in_dim=len(NUMERIC_COLS),
        hid=args.hid,
        design_dim_override=design_dim
    ).to(device)
    model = _reshape_main_net_to_ckpt(model, ckpt_model)
    model.load_state_dict(ckpt_model, strict=True)
    model.eval()

    # 8. 推理循环
    preds, gts = [], []

    # 准备常用 Tensor
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)

    print("[Info] Starting Inference...")
    with torch.no_grad():
        for xb, yb, cts in dl:
            xb = xb.to(device)
            # 标准化
            xb = (xb - x_mean_t) / x_std_t

            # 动态构建 batch 的 z
            # 根据每个样本的 cell_type (cts[i]) 查找对应的 z
            z_list = []
            for ct in cts:
                if ct in z_map:
                    z_list.append(z_map[ct])
                else:
                    # 如果遇到没见过的 cell，用 0 填充或报错
                    # 这里选择用 0 填充以防崩溃
                    z_list.append(torch.zeros(1, design_dim, device=device))

            zb = torch.cat(z_list, dim=0)  # [B, DesignDim]

            # 前向
            mu, logv, _, _ = model(xb, zb)

            # 反标
            mu_ps = (mu.cpu().numpy() * y_std) + y_mean
            preds.append(mu_ps)

            if yb is not None:
                gts.append(yb.numpy())

    # 9. 统计指标
    y_pred = np.concatenate(preds)

    if len(gts) > 0:
        y_true = np.concatenate(gts)
        # 简单过滤可能的离群点(可选)
        mae = np.mean(np.abs(y_pred - y_true))
        r2 = r2_score(y_true, y_pred)
        print("=" * 40)
        print(f"Target Test Result ({len(y_pred)} samples):")
        print(f"  MAE : {mae:.5f} (ps)")
        print(f"  R2  : {r2:.4f}")
        print("=" * 40)

        # 保存详细结果
        out_df = pd.DataFrame({"pred": y_pred, "label": y_true})
        out_df.to_csv(os.path.join(args.data_dir, "eval_result.csv"), index=False)
        print(f"[Info] Results saved to {os.path.join(args.data_dir, 'eval_result.csv')}")
    else:
        print(f"[Info] 无标签，已输出预测结果。")


if __name__ == "__main__":
    main()