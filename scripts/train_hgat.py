# === Python代码文件: train_hgat.py (最终完美版) ===

import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                collecting = True;
                buf.append(line)
        else:
            buf.append(line)
            if patt_end.match(line): break
    return "".join(buf) if buf else ""


# =========================================================
# Dataset
# =========================================================
class HGATDataset(Dataset):
    def __init__(self, csv_path, tech_label, x_mean, x_std, y_mean, y_std):
        self.df = pd.read_csv(csv_path, header=0)
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
        self.x_mean, self.x_std = x_mean.astype(np.float32), x_std.astype(np.float32)
        self.y_mean, self.y_std = np.float32(y_mean), np.float32(y_std)

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
# Graph Loading
# =========================================================
def load_all_graphs(data_dir, device, tgt_spice_override=None):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    graph_db, in_dim_map = {'SRC': {}, 'TGT': {}}, None

    print("[Info] Building SRC graphs...")
    for c_type, rel_path in meta["src_spi_by_cell"].items():
        full_path = rel_path if os.path.exists(rel_path) else os.path.join(data_dir, rel_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            devs = parse_transistors_spice(text)
            _, pins = parse_top_subckt_pins(text)
            if devs:
                g, feats, dims = build_dgl_graph_from_devs(devs, pins)
                graph_db['SRC'][c_type] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
                if in_dim_map is None: in_dim_map = dims
    print(f"  -> Loaded {len(graph_db['SRC'])} SRC graphs.")

    print("[Info] Building TGT graphs...")
    sp_file = meta.get("tgt_sp_file", "")
    if tgt_spice_override and os.path.exists(tgt_spice_override):
        sp_file = tgt_spice_override
    elif not os.path.exists(sp_file) and os.path.exists(os.path.join(data_dir, sp_file)):
        sp_file = os.path.join(data_dir, sp_file)

    if os.path.exists(sp_file):
        print(f"  -> Reading TGT SPICE from: {sp_file}")
        sp_text = open(sp_file, "r", encoding="utf-8", errors="ignore").read()
        for c_type, sub_name in meta["tgt_subckt_by_cell"].items():
            sub_txt = extract_subckt_text(sp_text, sub_name)
            if sub_txt:
                devs = parse_transistors_spice(sub_txt)
                _, pins = parse_top_subckt_pins(sub_txt)
                if devs:
                    g, feats, dims = build_dgl_graph_from_devs(devs, pins)
                    graph_db['TGT'][c_type] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
                    if in_dim_map is None: in_dim_map = dims
        print(f"  -> Loaded {len(graph_db['TGT'])} TGT graphs.")
    else:
        raise RuntimeError(f"CRITICAL: TGT SPICE file not found! ({sp_file})")

    return graph_db, in_dim_map, meta["cell_types"]


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--cmd_k", type=int, default=3)
    ap.add_argument("--lambda_cmd", type=float, default=1e-3)
    ap.add_argument("--lambda_supcon", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tgt_spice", type=str, default="", help="Path to ASAP7 SP file")
    ap.add_argument("--num_workers", type=int, default=0, help="Dataloader workers. Use >0 on Linux for speed.")
    ap.add_argument("--freeze_hgat", action="store_true", default=False)
    args = ap.parse_args()
    device = torch.device(args.device)

    # 1. Prepare Data and Scalers
    df_src = pd.read_csv(os.path.join(args.data_dir, "src_delay.csv"))
    df_tgt_l = pd.read_csv(os.path.join(args.data_dir, "tgt_delay_labeled.csv"))
    df_temp = pd.concat([df_src, df_tgt_l], ignore_index=True)

    if "pol_bit" not in df_temp.columns:
        if "pol" in df_temp.columns:
            df_temp["pol_bit"] = (df_temp["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df_temp["pol_bit"] = 0.0

    for c in NUMERIC_COLS_HGAT:
        if c not in df_temp.columns:
            df_temp[c] = 0.0

    x_all = df_temp[NUMERIC_COLS_HGAT].fillna(0.0).values.astype(np.float32)
    y_all = df_temp[[TARGET_COL]].values.astype(np.float32)
    x_mean, x_std = x_all.mean(0), x_all.std(0) + 1e-9
    y_mean, y_std = float(y_all.mean()), max(float(y_all.std()), 1e-2)
    print(f"[Info] y_mean = {y_mean:.6e}, raw_y_std = {float(y_all.std()):.6e}, used_y_std = {y_std:.6e}")

    scaler_stats = {"mean": {c: float(m) for c, m in zip(NUMERIC_COLS_HGAT, x_mean)},
                    "std": {c: float(s) for c, s in zip(NUMERIC_COLS_HGAT, x_std)}}
    json.dump(scaler_stats, open(os.path.join(args.data_dir, "scaler_stats.json"), "w"), indent=2)
    json.dump({"mean": y_mean, "std": y_std}, open(os.path.join(args.data_dir, "y_scaler.json"), "w"), indent=2)

    d_src = HGATDataset(os.path.join(args.data_dir, "src_delay.csv"), 0, x_mean, x_std, y_mean, y_std)
    d_tgt_l = HGATDataset(os.path.join(args.data_dir, "tgt_delay_labeled.csv"), 1, x_mean, x_std, y_mean, y_std)
    d_tgt_u = HGATDataset(os.path.join(args.data_dir, "tgt_delay_unlabeled.csv"), 1, x_mean, x_std, y_mean, y_std)
    l_src = DataLoader(d_src, args.batch, shuffle=True, drop_last=True, collate_fn=collate_fn,
                       num_workers=args.num_workers)
    l_tgt_l = DataLoader(d_tgt_l, args.batch, shuffle=True, drop_last=False, collate_fn=collate_fn,
                         num_workers=args.num_workers)
    l_tgt_u = DataLoader(d_tgt_u, args.batch, shuffle=True, drop_last=True, collate_fn=collate_fn,
                         num_workers=args.num_workers)

    # 2. Graph & Model
    graph_db, in_dim_map, cell_types_list = load_all_graphs(args.data_dir, device, args.tgt_spice)
    ctype_to_idx = {ct: i for i, ct in enumerate(cell_types_list)}
    enc = HGATDesignEncoder(in_dim_map, hid=64, out=64).to(device)
    model = DisentangledRegressor(len(NUMERIC_COLS_HGAT), hid=args.hid, design_dim_override=64).to(device)

    params_to_update = list(model.parameters())
    if not args.freeze_hgat:
        params_to_update.extend(list(enc.parameters()))
        print("[Info] HGAT encoder will be trained jointly.")
    else:
        for p in enc.parameters(): p.requires_grad = False
        print("[Info] HGAT encoder is FROZEN.")
    opt = torch.optim.Adam(params_to_update, lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=True)
    g_nll, supcon, best_nll = GaussianNLL(), SupConLoss(), 1e9

    # ==================== 【关键逻辑函数】 ====================
    def compute_basis(domain, encoder, graph_database, cell_types, device):
        # 必须使用 stack 保证 dim=2: [5, 64]，并且 squeeze() 确保每个元素是 1D
        vecs = [
            encoder(graph_database[domain][ct][0], graph_database[domain][ct][1]).squeeze()
            if ct in graph_database[domain]
            else torch.zeros(64, device=device)
            for ct in cell_types
        ]
        return torch.stack(vecs, dim=0)

    def map_z(cts, basis, ctype_map, dev):
        indices = torch.tensor([ctype_map[c] for c in cts], device=dev)
        return basis[indices]

    # ========================================================

    print(f"[Info] Start training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.freeze_hgat: enc.train()

        # 【逻辑分支】
        # 1. 如果冻结，只需要在 Epoch 开始算一次（无需梯度）
        if args.freeze_hgat:
            if 'z_basis_src' not in locals():
                with torch.no_grad():
                    z_basis_src = compute_basis('SRC', enc, graph_db, cell_types_list, device)
                    z_basis_tgt = compute_basis('TGT', enc, graph_db, cell_types_list, device)

        # 2. 如果联合训练，必须在 Batch 内部计算（见下文循环），这里不能预计算

        avg = {"loss": 0, "nll": 0, "cmd": 0}
        iters = max(len(l_src), len(l_tgt_u))
        iter_src, iter_tgt_l, iter_tgt_u = iter(l_src), iter(l_tgt_l), iter(l_tgt_u)

        for _ in range(iters):
            try:
                xs, ys, ls, cts_s = next(iter_src)
            except StopIteration:
                iter_src = iter(l_src); xs, ys, ls, cts_s = next(iter_src)
            try:
                xtl, ytl, ltl, cts_tl = next(iter_tgt_l)
            except StopIteration:
                iter_tgt_l = iter(l_tgt_l); xtl, ytl, ltl, cts_tl = next(iter_tgt_l)
            try:
                xtu, _, _, cts_tu = next(iter_tgt_u)
            except StopIteration:
                iter_tgt_u = iter(l_tgt_u); xtu, _, _, cts_tu = next(iter_tgt_u)

            xs, ys, xtl, ytl, xtu = xs.to(device), ys.to(device), xtl.to(device), ytl.to(device), xtu.to(device)
            opt.zero_grad()

            # 【联合训练核心】在每个 Step 内部计算，保证计算图完整
            if not args.freeze_hgat:
                z_basis_src = compute_basis('SRC', enc, graph_db, cell_types_list, device)
                z_basis_tgt = compute_basis('TGT', enc, graph_db, cell_types_list, device)

            zd_s = map_z(cts_s, z_basis_src, ctype_to_idx, device)
            zd_tl = map_z(cts_tl, z_basis_tgt, ctype_to_idx, device)
            zd_tu = map_z(cts_tu, z_basis_tgt, ctype_to_idx, device)

            mu_s, logv_s, zn_s, _ = model(xs, zd_s)
            mu_tl, logv_tl, zn_tl, _ = model(xtl, zd_tl)
            _, _, zn_tu, _ = model(xtu, zd_tu)
            loss_nll = g_nll(mu_s, logv_s, ys) + g_nll(mu_tl, logv_tl, ytl)
            loss_cmd = cmd_loss(zn_s, zn_tu, K=args.cmd_k) if args.lambda_cmd > 0 else torch.tensor(0.0, device=device)
            loss_sup = supcon(torch.cat([zn_s, zn_tl], dim=0), torch.cat([ls.to(device), ltl.to(device)],
                                                                         dim=0)) if args.lambda_supcon > 0 else torch.tensor(
                0.0, device=device)
            loss = loss_nll + args.lambda_cmd * loss_cmd + args.lambda_supcon * loss_sup

            check_finite("loss", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, 1.0)
            opt.step()
            avg["loss"] += loss.item();
            avg["nll"] += loss_nll.item();
            avg["cmd"] += loss_cmd.item()

        for k in avg: avg[k] /= iters
        scheduler.step(avg["nll"])
        print(f"[Epoch {epoch}] loss={avg['loss']:.4f} nll={avg['nll']:.4f} cmd={avg['cmd']:.4f}")

        if avg["nll"] < best_nll:
            best_nll = avg["nll"]
            state = {"model": model.state_dict(), "enc": enc.state_dict(), "hgat_in_dim_map": in_dim_map,
                     "design_dim": 64}
            torch.save(state, os.path.join(args.data_dir, "ckpt.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
