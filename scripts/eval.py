# eval.py  —— 直接整文件替换本脚本

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score

def _infer_hgat_hid_from_state(sd: dict) -> int:
    # 从 HGAT 编码器权重推断 hid（比如 embed.NET.weight 的形状 [hid, in_dim]）
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    # 兜底：readout.0.weight 的第二维通常等于 hid
    for k, v in sd.items():
        if k.startswith("readout.") and v.dim() == 2:
            return v.shape[1]
    raise RuntimeError("无法从 HGAT 编码器权重中推断 hid，请用 --hid 显式指定。")

def _reshape_main_net_to_ckpt(model, ckpt_model):
    """把 DisentangledRegressor 的层宽 reshape 成和 ckpt 一致（然后再 strict=True 加载权重）"""
    import torch.nn as nn

    # enc 的两层（错误信息里有 enc.0 / enc.2）
    enc0_out = ckpt_model["enc.0.weight"].shape[0]
    enc0_in  = ckpt_model["enc.0.weight"].shape[1]
    enc2_out = ckpt_model["enc.2.weight"].shape[0]
    enc2_in  = ckpt_model["enc.2.weight"].shape[1]
    model.enc[0] = nn.Linear(enc0_in, enc0_out)
    model.enc[2] = nn.Linear(enc2_in, enc2_out)

    # split_node（有些 ckpt 里会省略；有就按 ckpt 重建）
    if "split_node.weight" in ckpt_model:
        sn_out = ckpt_model["split_node.weight"].shape[0]
        sn_in  = ckpt_model["split_node.weight"].shape[1]
        model.split_node = nn.Linear(sn_in, sn_out)

    # head 两层（head.0 / head.2）
    h0_out = ckpt_model["head.0.weight"].shape[0]
    h0_in  = ckpt_model["head.0.weight"].shape[1]
    h2_out = ckpt_model["head.2.weight"].shape[0]
    h2_in  = ckpt_model["head.2.weight"].shape[1]
    model.head[0] = nn.Linear(h0_in, h0_out)
    model.head[2] = nn.Linear(h2_in, h2_out)

    # mu / log_var 的输入宽度
    mu_in = ckpt_model["mu.weight"].shape[1]
    model.mu = nn.Linear(mu_in, 1)
    model.log_var = nn.Linear(mu_in, 1)
    return model

# ====== 与训练保持一致的数值列 ======
NUMERIC_COLS = [
    "slew","cap","voltage","temp",
    "wp_over_wn","wp_sum","wn_sum",
    "is_inv","stack_pu","stack_pd","pol_bit"
]
TARGET_COL = "delay"

# ====== 模型 ======
from model import DisentangledRegressor  # 训练时用的同一个类

# ====== 可选：HGAT 编码器（只有 --use_hgat 才会 import） ======
def build_hgat_encoder_and_embed(spi_path, *, hid=None, ckpt_enc_state=None, device="cpu"):
    from hgat import HGATDesignEncoder, build_graph_from_spice

    # 1) 优先从 ckpt 的 HGAT 编码器权重自动推断 hid
    if ckpt_enc_state is not None:
        hid = _infer_hgat_hid_from_state(ckpt_enc_state)

    # 2) 没有 ckpt 或无法推断时，兜底到训练时默认的 64
    if hid is None:
        hid = 64

    # 3) 构图 + 特征
    g, feats, nfeat_dims = build_graph_from_spice(spi_path)

    # 4) 实例化并（如有）加载权重
    enc = HGATDesignEncoder(in_dim_map=nfeat_dims, hid=hid, out=64).to(device)
    if ckpt_enc_state is not None:
        enc.load_state_dict(ckpt_enc_state, strict=True)

    # 5) 前向
    feats = {k: v.to(device) for k, v in feats.items()}
    g = g.to(device)
    with torch.no_grad():
        z = enc(g, feats)

    # —— 新增（统一输出为 [B, D]）——
    if z.dim() == 1:
        z = z.unsqueeze(0)  # 变成 [1, D]
    elif z.dim() > 2:
        raise RuntimeError(f"HGAT encoder 返回了意外形状: {tuple(z.shape)}，期望是一维或二维张量。")

    return enc, z


# ====== 数据集 ======
class CSVDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"找不到数据集文件：{csv_path}")
        df = pd.read_csv(csv_path)

        # 生成 pol_bit（若已有则复用）
        if "pol_bit" not in df.columns:
            if "pol" in df.columns:
                # 兼容 'r'/'f' 或 'rise'/'fall'
                pol = df["pol"].astype(str).str.lower()
                df["pol_bit"] = pol.isin(["r","rise","1","true","t"]).astype(np.int32)
            else:
                df["pol_bit"] = 0

        # 缺失列补 0（避免 KeyError）
        for c in NUMERIC_COLS:
            if c not in df.columns:
                df[c] = 0.0

        self.x = df[NUMERIC_COLS].astype(np.float32).values
        if TARGET_COL in df.columns:
            self.y = df[TARGET_COL].astype(np.float32).values
        else:
            # 允许无标签推理（但无法计算 MAE/R2）
            self.y = None

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i])
        if self.y is None: return x, None
        return x, torch.tensor(self.y[i])

# ====== 读取 scaler ======
def load_scalers(data_dir:str):
    # 新方案
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")
    if os.path.isfile(ss_path) and os.path.isfile(ys_path):
        stats = json.load(open(ss_path,"r"))
        yinfo = json.load(open(ys_path,"r"))
        x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
        x_std  = np.array([stats["std"].get(c, 1.0)  for c in NUMERIC_COLS], dtype=np.float32)
        return x_mean, x_std, float(yinfo["mean"]), float(yinfo["std"])
    # 兼容旧的 scaler.json
    sc_path = os.path.join(data_dir, "scaler.json")
    if os.path.isfile(sc_path):
        sc = json.load(open(sc_path,"r"))
        x_mean = np.array(sc["x_mean"], dtype=np.float32)
        x_std  = np.array(sc["x_std"],  dtype=np.float32)
        return x_mean, x_std, float(sc["y_mean"]), float(sc["y_std"])
    raise FileNotFoundError("未找到 scaler_stats.json/y_scaler.json 或 scaler.json")

# ====== 解析参数 ======
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="训练/评测数据与 scaler 所在目录（例如 outputs）")
    ap.add_argument("--ckpt", required=True, help="训练时保存的权重路径（.pt）")
    ap.add_argument("--device", default="cpu")

    # —— 新增：与训练同构的选项（用于 HGAT）
    ap.add_argument("--use_hgat", action="store_true", help="与训练保持一致：是否启用 HGAT 设计向量")
    ap.add_argument("--hid", type=int, default=128, help="与训练保持一致的隐藏维度/设计向量维度")
    ap.add_argument("--tgt_spice", type=str, default="", help="目标工艺的 SPICE（HGAT 评测需要）")

    # 可选：指定评测 CSV，默认尝试 tgt_delay_labeled.csv
    ap.add_argument("--csv", type=str, default="", help="显式指定评测 CSV 文件路径")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 选择评测 CSV
    if args.csv:
        csv_path = args.csv
    else:
        cand = [
            os.path.join(args.data_dir, "tgt_delay_labeled.csv"),
            os.path.join(args.data_dir, "target_test.csv"),
            os.path.join(args.data_dir, "tgt.csv"),
        ]
        csv_path = next((p for p in cand if os.path.isfile(p)), None)
        if csv_path is None:
            raise FileNotFoundError("未找到默认的目标域评测 CSV，请用 --csv 指定。")

    # 读 scaler
    x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)

    # 数据
    ds  = CSVDataset(csv_path)
    dl  = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    # 读取/兼容 ckpt
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        ckpt_model = state["model"]
        ckpt_enc = state.get("enc", None)
        ckpt_inmap = state.get("hgat_in_dim_map", None)
        design_dim = int(state.get("design_dim", 0))
    else:
        ckpt_model = state
        ckpt_enc = None
        ckpt_inmap = None
        design_dim = (args.hid if args.use_hgat else 0)

    # HGAT：先算 z_tgt
    design_dim_from_ckpt = int(state.get("design_dim", 0))
    if args.use_hgat:
        # 先算 z_tgt
        _, z_tgt = build_hgat_encoder_and_embed(
            args.tgt_spice,
            ckpt_enc_state=ckpt_enc,
            device=device
        )

        z_tgt = z_tgt.to(device)
        print("[dbg] z_tgt.shape:", tuple(z_tgt.shape))
        print("[dbg] z_tgt per-dim std (mean):", float(z_tgt.std(dim=0).mean()))
        print("[dbg] z_tgt L2 mean:", float(z_tgt.norm(dim=-1).mean()))

        if z_tgt.dim() < 1:
            raise RuntimeError(f"z_tgt 形状异常: {tuple(z_tgt.shape)}")
        real_design_dim = int(z_tgt.shape[1])
        design_dim = real_design_dim if design_dim_from_ckpt <= 0 else design_dim_from_ckpt
    else:
        design_dim = None
    # 构建模型（外部 z）
    model = DisentangledRegressor(
        in_dim=len(NUMERIC_COLS),
        hid=args.hid,  # 主模型 MLP 的 hid（你训练用的是 128）
        design_dim_override=design_dim
    ).to(device)
    model = _reshape_main_net_to_ckpt(model, ckpt_model)
    model.load_state_dict(ckpt_model, strict=True)
    missing, unexpected = model.load_state_dict(ckpt_model, strict=False)
    print("[eval] missing:", missing)
    print("[eval] unexpected:", unexpected)
    # 不再检查 split_design.*（HGAT 采用外部 z，本来就没有这层）

    # 允许 BN/缓冲区的轻微不一致，但关键层必须存在
    # —— 原地替换 eval.py 中的这段检查逻辑 —— 参考位置在 load_state_dict 之后
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)

            # 标准化到训练分布
            xb = (xb - torch.from_numpy(x_mean).to(device)) / torch.from_numpy(x_std).to(device)

            if args.use_hgat:
                # 复制同一个设计向量到 batch
                zb = z_tgt.expand(xb.size(0), -1)
                mu, logv, _, _ = model(xb, zb)
            else:
                mu, logv, _, _ = model(xb)

            # 反标回 ps
            mu_ps = (mu.cpu().numpy() * y_std) + y_mean
            preds.append(torch.from_numpy(mu_ps))

            if yb is not None:
                gts.append(yb)

    y_pred = torch.cat(preds).numpy()
    if len(gts) == 0:
        # 无标签：仅导出预测
        out_csv = os.path.join(args.data_dir, "eval_pred.csv")
        pd.DataFrame({"pred_ps": y_pred}).to_csv(out_csv, index=False)
        print(f"[Info] 无 {TARGET_COL} 标签，已输出预测到 {out_csv}")
        return

    y_true = torch.cat(gts).numpy()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    r2  = float(r2_score(y_true, y_pred))
    print(f"Target Test: MAE={mae:.5f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()
