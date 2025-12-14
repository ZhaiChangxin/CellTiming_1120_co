import os
import json
import argparse
import re
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score

# 引入项目依赖（和训练脚本保持一致）
from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins

# ====== 数值列定义 (必须与 train_hgat.py 的 NUMERIC_COLS_HGAT 完全一致) ======
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


# ====== 辅助函数：从大 SPICE 文件中提取指定 subckt 文本 ======
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


# ====== 从 ckpt 里推断 HGAT 的 hid 宽度 ======
def _infer_hgat_hid_from_state(sd: dict) -> int:
    """
    根据 enc 的 state_dict 推断 GNN 隐藏维度 hid。
    训练脚本里 HGATDesignEncoder 是:
        embed.NET: Linear(in_dim, hid)
    所以 embed.NET.weight.shape = [hid, in_dim]
    """
    if sd is None:
        return 64
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    # 兜底
    for k, v in sd.items():
        if k.startswith("readout.") and v.dim() == 2:
            return v.shape[1]
    return 64


# ====== 根据 ckpt 动态调整 DisentangledRegressor 宽度 ======
def _reshape_main_net_to_ckpt(model, ckpt_model):
    """
    根据 ckpt 中 layer 的 weight shape 来调整 MLP 宽度。
    """
    import torch.nn as nn

    # 调整 Encoder MLP
    if "enc.0.weight" in ckpt_model:
        model.enc[0] = nn.Linear(
            ckpt_model["enc.0.weight"].shape[1],
            ckpt_model["enc.0.weight"].shape[0],
        )
    if "enc.2.weight" in ckpt_model:
        model.enc[2] = nn.Linear(
            ckpt_model["enc.2.weight"].shape[1],
            ckpt_model["enc.2.weight"].shape[0],
        )

    # 调整 Split Node
    if "split_node.weight" in ckpt_model:
        model.split_node = nn.Linear(
            ckpt_model["split_node.weight"].shape[1],
            ckpt_model["split_node.weight"].shape[0],
        )

    # 调整 Head
    if "head.0.weight" in ckpt_model:
        model.head[0] = nn.Linear(
            ckpt_model["head.0.weight"].shape[1],
            ckpt_model["head.0.weight"].shape[0],
        )
    if "head.2.weight" in ckpt_model:
        model.head[2] = nn.Linear(
            ckpt_model["head.2.weight"].shape[1],
            ckpt_model["head.2.weight"].shape[0],
        )

    # 调整 Mu/LogVar 输出头
    if "mu.weight" in ckpt_model:
        model.mu = nn.Linear(ckpt_model["mu.weight"].shape[1], 1)
    if "log_var.weight" in ckpt_model:
        model.log_var = nn.Linear(ckpt_model["log_var.weight"].shape[1], 1)

    return model


# ====== 评估用 Dataset（包含 cell_type） ======
class EvalDataset(Dataset):
    def __init__(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"找不到数据集文件：{csv_path}")
        df = pd.read_csv(csv_path)

        # 1. 构造 pol_bit（和训练逻辑保持一致）
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
        self.cell_types = df["cell_type"].values  # 关键：cell_type 决定设计向量 z

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
    return xs, ys, cts  # cts: tuple of str


# ====== 预计算目标域每个 cell_type 的设计向量 Z ======
def prepare_target_embeddings(data_dir, tgt_spice_path, enc, device):
    """
    根据 data_dir/meta.json 和 tgt_spice_path，
    为目标域的每种 cell_type 计算对应的 z 向量 (shape: [1, design_dim])。
    """
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json: {meta_path}，无法建立 cell_type -> subckt 映射。")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    tgt_map = meta.get("tgt_subckt_by_cell", {})
    if not tgt_map:
        raise RuntimeError("meta.json 中没有 'tgt_subckt_by_cell' 映射信息。")

    # 确定 SPICE 文件路径
    if not os.path.exists(tgt_spice_path):
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
            # 1. 提取 subckt 文本
            sub_txt = extract_subckt_text(sp_text, subckt_name)
            if not sub_txt:
                print(f"  [Warn] Subckt '{subckt_name}' (for {cell_type}) not found in SP file. Skipping.")
                continue

            # 2. 解析 & 构图
            devs = parse_transistors_spice(sub_txt)
            _, pins = parse_top_subckt_pins(sub_txt)
            if not devs:
                print(f"  [Warn] No transistor devs found for {cell_type}. Skipping.")
                continue

            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}

            # 3. 编码得到 z
            z = enc(g, feats)  # [D] or [D,]
            if z.dim() == 1:
                z = z.unsqueeze(0)  # [1, D]

            z_dict[cell_type] = z
            print(f"  -> {cell_type}: subckt='{subckt_name}', z.shape={tuple(z.shape)}")

    return z_dict


# ====== 预计算源域每个 cell_type 的设计向量 Z ======
def prepare_source_embeddings(data_dir, enc, device):
    """
    根据 data_dir/meta.json 的 src_spi_by_cell，
    为源域每种 cell_type 计算对应的 z 向量 (shape: [1, design_dim])。
    """
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 meta.json: {meta_path}，无法建立 cell_type -> SPICE 文件映射。")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    src_map = meta.get("src_spi_by_cell", {})
    if not src_map:
        raise RuntimeError("meta.json 中没有 'src_spi_by_cell' 映射信息。")

    z_dict = {}
    enc.eval()

    print("[Info] Pre-computing Z for source cells:")
    with torch.no_grad():
        for cell_type, rel_path in src_map.items():
            # 解析 SPICE 文件路径（支持相对 data_dir）
            if os.path.exists(rel_path):
                sp_path = rel_path
            else:
                sp_path = os.path.join(data_dir, rel_path)

            if not os.path.exists(sp_path):
                print(f"  [Warn] Source SPICE file for {cell_type} not found: {sp_path}. Skipping.")
                continue

            with open(sp_path, "r", encoding="utf-8", errors="ignore") as f_sp:
                sp_text = f_sp.read()

            devs = parse_transistors_spice(sp_text)
            _, pins = parse_top_subckt_pins(sp_text)
            if not devs:
                print(f"  [Warn] No transistor devs found for source {cell_type}. Skipping.")
                continue

            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}

            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)

            z_dict[cell_type] = z
            print(f"  -> {cell_type}: src_spice='{sp_path}', z.shape={tuple(z.shape)}")

    return z_dict


# ====== 通用评估函数（给定一个 csv 和对应的 z_map） ======
def run_eval_single(csv_path, z_map, tag, model, x_mean_t, x_std_t, y_mean, y_std, device, design_dim):
    print(f"[Info] Evaluating {tag} on: {csv_path}")
    ds = EvalDataset(csv_path)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=eval_collate)

    preds, gts = [], []

    with torch.no_grad():
        for xb, yb, cts in dl:
            xb = xb.to(device)
            xb = (xb - x_mean_t) / x_std_t  # 标准化

            # 为 batch 中每个样本构造对应的 z
            z_list = []
            for ct in cts:
                if ct in z_map:
                    z_list.append(z_map[ct])  # [1, D]
                else:
                    # 若遇到没在 meta 中出现过的 cell_type，用 0 向量兜底
                    z_list.append(torch.zeros(1, design_dim, device=device))

            zb = torch.cat(z_list, dim=0)  # [B, design_dim]

            mu, logv, _, _ = model(xb, zb)

            max_abs = 10.0  # 必须和 losses.py 中的 max_abs 一致
            mu = max_abs * torch.tanh(mu / max_abs)

            mu_np = mu.cpu().numpy()
            mu_ps = mu_np * y_std + y_mean
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
        print(f"{tag} Test Result ({len(y_pred)} samples):")
        print(f"  MAE : {mae:.5f} (ps)")
        print(f"  MSE : {mse:.5f} (ps^2)")
        print(f"  R2  : {r2:.4f}")
        print("=" * 40)

        out_df = pd.DataFrame({"pred": y_pred.flatten(), "label": y_true.flatten()})
    else:
        out_df = pd.DataFrame({"pred": y_pred.flatten()})

    out_name = f"eval_result_{tag.lower()}.csv"
    out_path = os.path.join(os.path.dirname(csv_path), out_name)
    out_df.to_csv(out_path, index=False)
    print(f"[Info] {tag} results saved to {out_path}")


# ====== 主流程 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="数据目录 (含 meta.json, scaler_stats.json 等)")
    ap.add_argument("--ckpt", required=True, help="训练保存的 ckpt.pt 路径")
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--use_hgat", action="store_true", default=True, help="保留以兼容参数，不实际控制逻辑")
    ap.add_argument("--hid", type=int, default=128, help="MLP 隐藏层维度（会被 ckpt reshape 覆盖）")
    ap.add_argument("--tgt_spice", type=str, default="", help="ASAP7 SPICE 文件路径")
    ap.add_argument("--csv", type=str, default="", help="若指定，则只评估该 CSV（按目标域处理）")

    args = ap.parse_args()
    device = torch.device(args.device)

    # 1. 读取 scaler（必须和训练时一致）
    ss_path = os.path.join(args.data_dir, "scaler_stats.json")
    ys_path = os.path.join(args.data_dir, "y_scaler.json")
    if os.path.exists(ss_path) and os.path.exists(ys_path):
        stats = json.load(open(ss_path, "r"))
        yinfo = json.load(open(ys_path, "r"))
        x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
        x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
        y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    else:
        raise FileNotFoundError("找不到 scaler_stats.json / y_scaler.json，无法反归一化。")

    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)

    # 2. 加载 ckpt
    # PyTorch 2.x 引入了 `weights_only=True` 的“安全反序列化”，
    # 但很多旧 ckpt 里会保存 numpy 对象（例如 ndarray），会触发 UnpicklingError。
    # 这里做一个「尽量安全」的兼容：
    #   1) 先尝试 allowlist numpy 的 _reconstruct（只影响 weights_only=True 的白名单）
    #   2) 再尝试 weights_only=True
    #   3) 若仍失败，并且 ckpt 来自你自己/可信来源，则降级 weights_only=False（存在任意代码执行风险！）
    try:
        from numpy.core.multiarray import _reconstruct  # type: ignore
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([_reconstruct])
    except Exception:
        # 不影响后续加载逻辑
        pass

    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
    except pickle.UnpicklingError as e:
        print("[Warn] torch.load(weights_only=True) failed:")
        print(f"       {e}")
        print("[Warn] Falling back to torch.load(weights_only=False).")
        print("       ⚠️  仅当 ckpt 来自你自己/可信来源时才这样做（否则可能有任意代码执行风险）。")
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        ckpt_model = state["model"]
        ckpt_enc = state.get("enc", None)
        ckpt_inmap = state.get("hgat_in_dim_map", None)
        design_dim = int(state.get("design_dim", 64))
    else:
        raise RuntimeError("ckpt 格式不符合预期：需要包含 'model' 和 'enc' 等键。")

    # 3. 初始化 HGAT encoder，并加载权重
    if ckpt_inmap is None:
        print("[Warn] ckpt 中没有 'hgat_in_dim_map'，使用默认 {'NET':4,'PMOS':2,'NMOS':2} 尝试加载。")
        ckpt_inmap = {"NET": 4, "PMOS": 2, "NMOS": 2}

    enc_hid = _infer_hgat_hid_from_state(ckpt_enc)
    enc = HGATDesignEncoder(in_dim_map=ckpt_inmap, hid=enc_hid, out=design_dim).to(device)
    enc.load_state_dict(ckpt_enc, strict=True)

    # 4. 初始化 DisentangledRegressor，并加载权重
    model = DisentangledRegressor(
        in_dim=len(NUMERIC_COLS),
        hid=args.hid,
        design_dim_override=design_dim,
    )

    model = _reshape_main_net_to_ckpt(model, ckpt_model)
    model.load_state_dict(ckpt_model, strict=True)
    model = model.to(device)
    model.eval()

    # 5. 判断评估哪些数据集
    # 定义标准路径
    src_csv_default = os.path.join(args.data_dir, "src_delay.csv")
    tgt_train_default = os.path.join(args.data_dir, "tgt_train.csv")
    tgt_val_default = os.path.join(args.data_dir, "tgt_val.csv")
    tgt_test_default = os.path.join(args.data_dir, "tgt_test.csv")

    eval_src = False
    eval_tgt_train = False
    eval_tgt_val = False
    eval_tgt_test = False
    eval_tgt_custom = False

    custom_tgt_csv = None

    if args.csv:
        # 用户手动指定 CSV：只评估这一份（按目标域处理）
        if not os.path.isfile(args.csv):
            raise FileNotFoundError(f"--csv 指定的文件不存在: {args.csv}")
        eval_tgt_custom = True
        custom_tgt_csv = args.csv
    else:
        # 自动检测源域 / 目标域(Train/Val/Test)
        if os.path.isfile(src_csv_default):
            eval_src = True

        # 依次检测目标域各集
        if os.path.isfile(tgt_train_default):
            eval_tgt_train = True
        if os.path.isfile(tgt_val_default):
            eval_tgt_val = True
        if os.path.isfile(tgt_test_default):
            eval_tgt_test = True

    if not any([eval_src, eval_tgt_train, eval_tgt_val, eval_tgt_test, eval_tgt_custom]):
        raise RuntimeError("未在 data_dir 找到任何标准数据集 (src_delay, tgt_train, tgt_val, tgt_test)，也未指定 --csv。")

    # 6. 预计算源域 / 目标域的设计向量 z
    #    目标域的 embed 只需算一次，可被 Train/Val/Test 共用
    z_src = None
    z_tgt = None

    if eval_src:
        z_src = prepare_source_embeddings(args.data_dir, enc, device)

    # 只要涉及任意目标域数据，就需要准备 z_tgt
    need_tgt_embed = (eval_tgt_train or eval_tgt_val or eval_tgt_test or eval_tgt_custom)
    if need_tgt_embed:
        z_tgt = prepare_target_embeddings(args.data_dir, args.tgt_spice, enc, device)

    # 7. 分别跑评估

    # (A) 源域
    if eval_src:
        run_eval_single(
            csv_path=src_csv_default,
            z_map=z_src,
            tag="Source",
            model=model,
            x_mean_t=x_mean_t,
            x_std_t=x_std_t,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            design_dim=design_dim,
        )

    # (B) 目标域 - 训练集
    if eval_tgt_train:
        run_eval_single(
            csv_path=tgt_train_default,
            z_map=z_tgt,
            tag="Target_Train",
            model=model,
            x_mean_t=x_mean_t,
            x_std_t=x_std_t,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            design_dim=design_dim,
        )

    # (C) 目标域 - 验证集
    if eval_tgt_val:
        run_eval_single(
            csv_path=tgt_val_default,
            z_map=z_tgt,
            tag="Target_Val",
            model=model,
            x_mean_t=x_mean_t,
            x_std_t=x_std_t,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            design_dim=design_dim,
        )

    # (D) 目标域 - 测试集
    if eval_tgt_test:
        run_eval_single(
            csv_path=tgt_test_default,
            z_map=z_tgt,
            tag="Target_Test",
            model=model,
            x_mean_t=x_mean_t,
            x_std_t=x_std_t,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            design_dim=design_dim,
        )

    # (E) 目标域 - 用户自定义
    if eval_tgt_custom:
        run_eval_single(
            csv_path=custom_tgt_csv,
            z_map=z_tgt,
            tag="Target_Custom",
            model=model,
            x_mean_t=x_mean_t,
            x_std_t=x_std_t,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            design_dim=design_dim,
        )


if __name__ == "__main__":
    main()
