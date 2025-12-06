import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 引用现有项目文件
from model import DisentangledRegressor
from losses import GaussianNLL, cmd_loss, SupConLoss
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs

# =========================================================
# 配置
# =========================================================
NUMERIC_COLS_HGAT = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv", "stack_pu", "stack_pd",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit"
]
TARGET_COL = "delay"


def check_finite(name, t):
    if t is None: return
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


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


# =========================================================
# Dataset
# =========================================================
class HGATDataset(Dataset):
    def __init__(self, csv_path, tech_label, x_mean, x_std, y_mean, y_std):
        self.df = pd.read_csv(csv_path, header=0)

        # 预处理
        if "pol_bit" not in self.df.columns:
            if "pol" in self.df.columns:
                self.df["pol_bit"] = (self.df["pol"].astype(str) == "rise").astype(np.float32)
            else:
                self.df["pol_bit"] = 0.0

        for c in NUMERIC_COLS_HGAT:
            if c not in self.df.columns: self.df[c] = 0.0

        self.x = self.df[NUMERIC_COLS_HGAT].fillna(0.0).values.astype(np.float32)
        self.y = self.df[TARGET_COL].values.astype(np.float32)
        self.cell_types = self.df["cell_type"].values
        self.labels = np.full((len(self.df),), tech_label, dtype=np.int64)

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = np.float32(y_mean)
        self.y_std = np.float32(y_std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x_norm = (self.x[i] - self.x_mean) / self.x_std
        y_norm = (self.y[i] - self.y_mean) / self.y_std
        return x_norm, y_norm, self.labels[i], self.cell_types[i]


def collate_fn(batch):
    xs, ys, ls, cts = zip(*batch)
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)), torch.tensor(np.stack(ls)), cts


# =========================================================
# Graph Loading (Fixed)
# =========================================================
def load_all_graphs(data_dir, device, tgt_spice_override=None):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    graph_db = {'SRC': {}, 'TGT': {}}
    in_dim_map = None

    # --- SRC ---
    print("[Info] Building SRC graphs...")
    for c_type, rel_path in meta["src_spi_by_cell"].items():
        full_path = rel_path
        if not os.path.exists(full_path):
            full_path = os.path.join(data_dir, rel_path)

        if os.path.exists(full_path):
            txt = open(full_path, "r", encoding="utf-8", errors="ignore").read()
            devs = parse_transistors_spice(txt)
            _, pins = parse_top_subckt_pins(txt)
            if devs:
                g, feats, dims = build_dgl_graph_from_devs(devs, pins)
                graph_db['SRC'][c_type] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
                if in_dim_map is None: in_dim_map = dims

    print(f"  -> Loaded {len(graph_db['SRC'])} SRC graphs.")

    # --- TGT (Crucial Fix) ---
    print("[Info] Building TGT graphs...")
    sp_file = meta.get("tgt_sp_file", "")

    # 优先使用命令行参数覆盖
    if tgt_spice_override and os.path.exists(tgt_spice_override):
        sp_file = tgt_spice_override
    elif not os.path.exists(sp_file):
        # 尝试相对路径拼接
        cand = os.path.join(data_dir, sp_file)
        if os.path.exists(cand): sp_file = cand

    if os.path.exists(sp_file):
        print(f"  -> Reading TGT SPICE from: {sp_file}")
        sp_text = open(sp_file, "r", encoding="utf-8", errors="ignore").read()
        count = 0
        for c_type, sub_name in meta["tgt_subckt_by_cell"].items():
            sub_txt = extract_subckt_text(sp_text, sub_name)
            if sub_txt:
                devs = parse_transistors_spice(sub_txt)
                _, pins = parse_top_subckt_pins(sub_txt)
                if devs:
                    g, feats, dims = build_dgl_graph_from_devs(devs, pins)
                    graph_db['TGT'][c_type] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
                    if in_dim_map is None: in_dim_map = dims
                    count += 1
        print(f"  -> Loaded {count} TGT graphs.")
    else:
        print(f"  [ERROR] TGT SPICE file not found! Looked at: {sp_file}")

    # 严格检查：如果没加载到 TGT 图，必须报错，否则训练无效
    if len(graph_db['TGT']) == 0:
        raise RuntimeError("CRITICAL: No TGT graphs loaded. Please check --tgt_spice path.")

    return graph_db, in_dim_map, meta["cell_types"]


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--cmd_k", type=int, default=3)
    ap.add_argument("--lambda_cmd", type=float, default=1e-3)
    ap.add_argument("--lambda_supcon", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 新增：强制指定 ASAP7 路径
    ap.add_argument("--tgt_spice", type=str, default="", help="Path to ASAP7 SP file")
    ap.add_argument("--use_hgat", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1. Prepare Data
    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    tgt_l_csv = os.path.join(args.data_dir, "tgt_delay_labeled.csv")
    tgt_u_csv = os.path.join(args.data_dir, "tgt_delay_unlabeled.csv")

    df_src = pd.read_csv(src_csv)
    df_tgt_l = pd.read_csv(tgt_l_csv)

    # Simple concat for fit
    df_temp = pd.concat([df_src, df_tgt_l], ignore_index=True)
    for c in NUMERIC_COLS_HGAT:
        if c not in df_temp.columns: df_temp[c] = 0.0

    x_all = df_temp[NUMERIC_COLS_HGAT].fillna(0.0).values.astype(np.float32)
    y_all = df_temp[[TARGET_COL]].values.astype(np.float32)

    x_mean, x_std = x_all.mean(0), x_all.std(0) + 1e-9
    y_mean, y_std = float(y_all.mean()), float(y_all.std()) + 1e-9

    # Save scalers
    scaler_stats = {
        "mean": {c: float(m) for c, m in zip(NUMERIC_COLS_HGAT, x_mean)},
        "std": {c: float(s) for c, s in zip(NUMERIC_COLS_HGAT, x_std)}
    }
    json.dump(scaler_stats, open(os.path.join(args.data_dir, "scaler_stats.json"), "w"), indent=2)
    json.dump({"mean": y_mean, "std": y_std}, open(os.path.join(args.data_dir, "y_scaler.json"), "w"), indent=2)

    # Loaders
    d_src = HGATDataset(src_csv, 0, x_mean, x_std, y_mean, y_std)
    d_tgt_l = HGATDataset(tgt_l_csv, 1, x_mean, x_std, y_mean, y_std)
    d_tgt_u = HGATDataset(tgt_u_csv, 1, x_mean, x_std, y_mean, y_std)

    l_src = DataLoader(d_src, args.batch, shuffle=True, drop_last=True, collate_fn=collate_fn)
    l_tgt_l = DataLoader(d_tgt_l, args.batch, shuffle=True, drop_last=False, collate_fn=collate_fn)
    l_tgt_u = DataLoader(d_tgt_u, args.batch, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # 2. Graph & Model
    # 传递 tgt_spice 路径
    graph_db, in_dim_map, cell_types_list = load_all_graphs(args.data_dir, device, args.tgt_spice)
    ctype_to_idx = {ct: i for i, ct in enumerate(cell_types_list)}

    enc = HGATDesignEncoder(in_dim_map, hid=64, out=64).to(device)
    model = DisentangledRegressor(len(NUMERIC_COLS_HGAT), hid=args.hid, design_dim_override=64).to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(enc.parameters()), lr=args.lr)
    g_nll = GaussianNLL()
    supcon = SupConLoss()

    # 3. Training
    best_nll = 1e9

    # 缓存 Basis 计算 (优化速度)
    def compute_basis(domain_key):
        vecs = []
        for ct in cell_types_list:
            if ct in graph_db[domain_key]:
                g, f = graph_db[domain_key][ct]
                z = enc(g, f)
                if z.dim() == 1: z = z.unsqueeze(0)
                vecs.append(z)
            else:
                # 这种情况下应该报警，但为了代码健壮性暂时保留
                vecs.append(torch.zeros(1, 64, device=device))
        return torch.cat(vecs, dim=0)

    print(f"[Info] Start training...")
    for epoch in range(1, args.epochs + 1):
        model.train();
        enc.train()
        avg = {"loss": 0, "nll": 0, "cmd": 0}
        iters = max(len(l_src), len(l_tgt_u))

        iter_src = iter(l_src)
        iter_tgt_l = iter(l_tgt_l)
        iter_tgt_u = iter(l_tgt_u)

        for _ in range(iters):
            try:
                xs, ys, ls, cts_s = next(iter_src)
            except:
                iter_src = iter(l_src); xs, ys, ls, cts_s = next(iter_src)
            try:
                xtl, ytl, ltl, cts_tl = next(iter_tgt_l)
            except:
                iter_tgt_l = iter(l_tgt_l); xtl, ytl, ltl, cts_tl = next(iter_tgt_l)
            try:
                xtu, ytu, ltu, cts_tu = next(iter_tgt_u)
            except:
                iter_tgt_u = iter(l_tgt_u); xtu, ytu, ltu, cts_tu = next(iter_tgt_u)

            xs, ys = xs.to(device), ys.to(device)
            xtl, ytl = xtl.to(device), ytl.to(device)
            xtu = xtu.to(device)

            opt.zero_grad()

            # 计算当前 Z
            z_basis_src = compute_basis('SRC')
            z_basis_tgt = compute_basis('TGT')

            # 映射
            def map_z(cts, basis):
                indices = [ctype_to_idx[c] for c in cts]
                return basis[torch.tensor(indices, device=device)]

            zd_s = map_z(cts_s, z_basis_src)
            zd_tl = map_z(cts_tl, z_basis_tgt)
            zd_tu = map_z(cts_tu, z_basis_tgt)

            # Forward
            mu_s, logv_s, zn_s, _ = model(xs, zd_s)
            mu_tl, logv_tl, zn_tl, _ = model(xtl, zd_tl)
            _, _, _, _ = model(xtu, zd_tu)

            # Loss
            loss_nll = g_nll(mu_s, logv_s, ys) + g_nll(mu_tl, logv_tl, ytl)
            loss_cmd = cmd_loss(zd_s, zd_tu, K=args.cmd_k) if args.lambda_cmd > 0 else torch.tensor(0.)

            # SupCon (可选)
            loss_sup = torch.tensor(0.)
            if args.lambda_supcon > 0:
                zn = torch.cat([zn_s, zn_tl], dim=0)
                labs = torch.cat([ls.to(device), ltl.to(device)], dim=0)
                loss_sup = supcon(zn, labs)

            loss = loss_nll + args.lambda_cmd * loss_cmd + args.lambda_supcon * loss_sup

            check_finite("loss", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            avg["loss"] += loss.item()
            avg["nll"] += loss_nll.item()
            avg["cmd"] += loss_cmd.item()

        for k in avg: avg[k] /= iters
        print(f"[Epoch {epoch}] loss={avg['loss']:.4f} nll={avg['nll']:.4f} cmd={avg['cmd']:.4f}")

        if avg["nll"] < best_nll:
            best_nll = avg["nll"]
            state = {
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_dim_map,
                "design_dim": 64
            }
            torch.save(state, os.path.join(args.data_dir, "ckpt.pt"))

    print("Done.")


if __name__ == "__main__":
    main()