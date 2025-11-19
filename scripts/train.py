
import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import DisentangledRegressor
from losses import GaussianNLL, cmd_loss, SupConLoss
from spi2graph import parse_transistors_spice, extract_wl_features, parse_top_subckt_pins

NUMERIC_COLS_BASE = ["slew","cap","voltage","temp","wp_over_wn","wp_sum","wn_sum","is_inv","stack_pu","stack_pd","pol_bit"]
TARGET_COL = "delay"

def build_dgl_graph_from_devs(devs, top_pins):
    import dgl, torch
    nets = {}
    def net_id(n):
        if n not in nets: nets[n] = len(nets)
        return nets[n]

    p_count=n_count=0
    gate_src_p= []; gate_dst_p= []
    gate_src_n= []; gate_dst_n= []
    sd_src_p=   []; sd_dst_p=   []
    sd_src_n=   []; sd_dst_n=   []

    for d in devs:
        if d["type"].startswith("p"):
            mid = p_count; p_count += 1
            gate_src_p.append(net_id(d["g"])); gate_dst_p.append(mid)
            sd_src_p.extend([mid, mid]); sd_dst_p.extend([net_id(d["s"]), net_id(d["d"])])
        else:
            mid = n_count; n_count += 1
            gate_src_n.append(net_id(d["g"])); gate_dst_n.append(mid)
            sd_src_n.extend([mid, mid]); sd_dst_n.extend([net_id(d["s"]), net_id(d["d"])])

    data_dict = {}
    if p_count > 0:
        data_dict[('NET','gate_of','PMOS')] = (torch.tensor(gate_src_p), torch.tensor(gate_dst_p))
        data_dict[('PMOS','sd_to','NET')]   = (torch.tensor(sd_src_p),  torch.tensor(sd_dst_p))
        data_dict[('NET','back_sd','PMOS')] = (torch.tensor(sd_dst_p),  torch.tensor(sd_src_p))
    if n_count > 0:
        data_dict[('NET','gate_of','NMOS')] = (torch.tensor(gate_src_n), torch.tensor(gate_dst_n))
        data_dict[('NMOS','sd_to','NET')]   = (torch.tensor(sd_src_n),  torch.tensor(sd_dst_n))
        data_dict[('NET','back_sd','NMOS')] = (torch.tensor(sd_dst_n),  torch.tensor(sd_src_n))
    g = dgl.heterograph(data_dict, num_nodes_dict={'NET': len(nets), 'PMOS': p_count, 'NMOS': n_count})

    import numpy as np
    f_net = []
    for name, nid in sorted(nets.items(), key=lambda x:x[1]):
        is_vdd = 1.0 if name.upper()=="VDD" else 0.0
        is_vss = 1.0 if name.upper()=="VSS" else 0.0
        is_a   = 1.0 if name.upper()=="A" else 0.0
        is_y   = 1.0 if name.upper()=="Y" or name.upper()=="ZN" else 0.0
        f_net.append([is_vdd,is_vss,is_a,is_y])
    f_net = torch.tensor(np.array(f_net, dtype=np.float32))
    def mos_feats(list_dev):
        arr = []
        for d in list_dev:
            W = d["W"] if d["W"] is not None else 0.0
            L = d["L"] if d["L"] is not None else 0.0
            arr.append([W, L])
        if len(arr)==0:
            return torch.zeros((0,2), dtype=torch.float32)
        return torch.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats([d for d in devs if d["type"].startswith("p")])
    f_n = mos_feats([d for d in devs if d["type"].startswith("n")])
    feats = {'NET': f_net, 'PMOS': f_p, 'NMOS': f_n}
    in_dim_map = {'NET': f_net.shape[1] if f_net.numel() else 4, 'PMOS': 2, 'NMOS': 2}
    return g, feats, in_dim_map

class CSVDataset(Dataset):
    def __init__(self, csv_path, tech_label, x_mean, x_std, y_mean, y_std):
        self.df = pd.read_csv(csv_path)
        self.df["pol_bit"] = (self.df["pol"].astype(str)=="rise").astype(np.float32)
        self.df["is_inv"] = 1.0; self.df["stack_pu"]=1.0; self.df["stack_pd"]=1.0

        for c in NUMERIC_COLS_BASE:
            if c not in self.df.columns: self.df[c] = np.nan
        self.df[NUMERIC_COLS_BASE] = self.df[NUMERIC_COLS_BASE].fillna(0.0).astype(np.float32)

        self.x = self.df[NUMERIC_COLS_BASE].values.astype(np.float32)
        self.y = self.df[TARGET_COL].values.astype(np.float32)

        # scalers
        self.x_mean = x_mean.astype(np.float32)
        self.x_std  = x_std.astype(np.float32)
        self.y_mean = np.float32(y_mean)
        self.y_std  = np.float32(y_std)

        self.labels = np.full((len(self.df),), tech_label, dtype=np.int64)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        x = (self.x[i] - self.x_mean) / self.x_std
        y = (self.y[i] - self.y_mean) / self.y_std
        return x, y, self.labels[i]


def collate(batch):
    xs, ys, ls = zip(*batch)
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)), torch.tensor(np.stack(ls))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--cmd_k", type=int, default=5)
    ap.add_argument("--lambda_cmd", type=float, default=0.1)
    ap.add_argument("--lambda_supcon", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_hgat", action="store_true")
    ap.add_argument("--src_spice", default="")
    ap.add_argument("--tgt_spice", default="")
    args = ap.parse_args()

    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    tgt_l_csv = os.path.join(args.data_dir, "tgt_delay_labeled.csv")
    tgt_u_csv = os.path.join(args.data_dir, "tgt_delay_unlabeled.csv")
    NUMERIC_COLS_BASE = ["slew", "cap", "voltage", "temp", "wp_over_wn", "wp_sum", "wn_sum", "is_inv", "stack_pu",
                         "stack_pd", "pol_bit"]
    TARGET_COL = "delay"

    df_src_full = pd.read_csv(src_csv)
    df_tgt_l_full = pd.read_csv(tgt_l_csv)
    # === ensure all feature columns exist and build pol_bit ===
    def ensure_cols(df):
        import numpy as np
        # 构造 pol_bit：rise=1.0, else=0.0
        if "pol_bit" not in df.columns:
            if "pol" in df.columns:
                df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
            else:
                df["pol_bit"] = 0.0

        # 这些列是训练用的固定输入列
        NUMERIC_COLS_BASE = ["slew","cap","voltage","temp",
                             "wp_over_wn","wp_sum","wn_sum",
                             "is_inv","stack_pu","stack_pd","pol_bit"]

        # 缺了就补 0.0，保证可索引
        for c in NUMERIC_COLS_BASE:
            if c not in df.columns:
                df[c] = 0.0

        # 填 NaN 并转成 float32
        df[NUMERIC_COLS_BASE] = df[NUMERIC_COLS_BASE].fillna(0.0).astype(np.float32)
        return df

    df_src_full = ensure_cols(df_src_full)
    df_tgt_l_full = ensure_cols(df_tgt_l_full)

    x_all = pd.concat([df_src_full[NUMERIC_COLS_BASE], df_tgt_l_full[NUMERIC_COLS_BASE]], axis=0).astype(np.float32)
    y_all = pd.concat([df_src_full[[TARGET_COL]], df_tgt_l_full[[TARGET_COL]]], axis=0).astype(np.float32)

    x_mean = x_all.mean(0).values
    x_std = (x_all.std(0).values + 1e-9)
    y_mean = float(y_all.mean().values[0])
    y_std = float(y_all.std().values[0] + 1e-9)

    scaler = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean, "y_std": y_std}
    json.dump(scaler, open(os.path.join(args.data_dir, "scaler.json"), "w"))
    json.dump({"mean": {c: float(m) for c, m in zip(NUMERIC_COLS_BASE, x_mean)},
               "std": {c: float(s) for c, s in zip(NUMERIC_COLS_BASE, x_std)}},
              open(os.path.join(args.data_dir, "scaler_stats.json"), "w"), indent=2)
    json.dump({"mean": float(y_mean), "std": float(y_std)},
              open(os.path.join(args.data_dir, "y_scaler.json"), "w"), indent=2)
    #d_src = CSVDataset(src_csv, tech_label=0)
    #d_tgt_l = CSVDataset(tgt_l_csv, tech_label=1)
    #d_tgt_u = CSVDataset(tgt_u_csv, tech_label=1)
    d_src   = CSVDataset(src_csv,   tech_label=0, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    d_tgt_l = CSVDataset(tgt_l_csv, tech_label=1, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    d_tgt_u = CSVDataset(tgt_u_csv, tech_label=1, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    loader_src = DataLoader(d_src, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)
    #loader_tgt_l = DataLoader(d_tgt_l, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)
    loader_tgt_l = DataLoader(d_tgt_l, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=collate)
    loader_tgt_u = DataLoader(d_tgt_u, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)

    design_dim_override = None
    if args.use_hgat:
        from hgat import HGATDesignEncoder
        text_src = open(args.src_spice,"r",encoding="utf-8",errors="ignore").read() if args.src_spice and os.path.exists(args.src_spice) else ""
        text_tgt = open(args.tgt_spice,"r",encoding="utf-8",errors="ignore").read() if args.tgt_spice and os.path.exists(args.tgt_spice) else ""
        devs_src = parse_transistors_spice(text_src) if text_src else []
        devs_tgt = parse_transistors_spice(text_tgt) if text_tgt else []
        name_src, pins_src = parse_top_subckt_pins(text_src) if text_src else (None, [])
        name_tgt, pins_tgt = parse_top_subckt_pins(text_tgt) if text_tgt else (None, [])
        bundle_src = build_dgl_graph_from_devs(devs_src, pins_src) if devs_src else None
        bundle_tgt = build_dgl_graph_from_devs(devs_tgt, pins_tgt) if devs_tgt else None
        if bundle_src or bundle_tgt:
            # initialize one encoder; shared weights across src/tgt
            in_map = (bundle_src or bundle_tgt)[2]
            enc = HGATDesignEncoder(in_map, hid=64, out=64, num_heads=1).to(args.device)
            design_dim_override = 64
        else:
            enc = None
    else:
        enc = None

    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS_BASE), hid=args.hid, design_dim_override=design_dim_override).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    #opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 余弦退火（整轮退火到 0.1× 初始学习率）
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    g_nll = GaussianNLL()
    #g_nll = GaussianNLL(sigma_floor=0.05, sigma_ceiling=3.0)
    supcon = SupConLoss()

    def compute_z(bundle):
        if (not args.use_hgat) or (bundle is None) or (enc is None):
            return None
        g, feats, in_map = bundle
        g = g.to(args.device)
        feats = {k: v.to(args.device) for k, v in feats.items()}
        #return enc(g, feats)
        with torch.no_grad():
            z = enc(g, feats)
        return z.detach()

    def run_epoch(epoch):
        model.train()
        #iters = min(len(loader_src), len(loader_tgt_u))
        iters = max(len(loader_src), len(loader_tgt_u))
        avg = {"loss":0, "nll":0, "cmd":0, "supcon":0}
        src_it = iter(loader_src)
        tgt_l_it = iter(loader_tgt_l)
        tgt_u_it = iter(loader_tgt_u)

        z_src = compute_z(bundle_src) if args.use_hgat else None
        z_tgt = compute_z(bundle_tgt) if args.use_hgat else None

        for _ in range(iters):
            try: xs, ys, ls = next(src_it)
            except StopIteration: src_it = iter(loader_src); xs, ys, ls = next(src_it)
            try: xtl, ytl, ltl = next(tgt_l_it)
            except StopIteration: tgt_l_it = iter(loader_tgt_l); xtl, ytl, ltl = next(tgt_l_it)
            try: xtu, ytu, ltu = next(tgt_u_it)
            except StopIteration: tgt_u_it = iter(loader_tgt_u); xtu, ytu, ltu = next(tgt_u_it)

            xs, ys, ls = xs.to(args.device), ys.to(args.device), ls.to(args.device)
            xtl, ytl, ltl = xtl.to(args.device), ytl.to(args.device), ltl.to(args.device)
            xtu, ytu, ltu = xtu.to(args.device), ytu.to(args.device), ltu.to(args.device)

            opt.zero_grad()
            mu_s, logv_s, zn_s, zd_s = model(xs, z_src)
            mu_tl, logv_tl, zn_tl, zd_tl = model(xtl, z_tgt)
            nll = g_nll(mu_s, logv_s, ys) + g_nll(mu_tl, logv_tl, ytl)
            with torch.no_grad():
                _, _, _, zd_u = model(xtu, z_tgt)
            cmd = cmd_loss(zd_s, zd_u, K=args.cmd_k)
            zn = torch.cat([zn_s, zn_tl], dim=0)
            labs = torch.cat([ls, ltl], dim=0)
            s_con = supcon(zn, labs)
            loss = nll + args.lambda_cmd * cmd + args.lambda_supcon * s_con
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            avg["loss"] += loss.item(); avg["nll"] += nll.item(); avg["cmd"] += cmd.item(); avg["supcon"] += s_con.item()
        for k in avg: avg[k] /= iters
        print(f"[Epoch {epoch}] loss={avg['loss']:.4f} nll={avg['nll']:.4f} cmd={avg['cmd']:.4f} supcon={avg['supcon']:.4f}")
        #scheduler.step()
        return avg

    best = 1e9
    best_state = None
    for ep in range(1, args.epochs + 1):
        stats = run_epoch(ep)
        if stats["nll"] < best:
            best = stats["nll"]
            best_state = {
                "model": model.state_dict(),
                "enc": (enc.state_dict() if enc is not None else None),
                "hgat_in_dim_map": (in_map if enc is not None else None),
                "design_dim": (design_dim_override if design_dim_override is not None else 0),
                "numeric_cols": NUMERIC_COLS_BASE,
            }
    torch.save(best_state, os.path.join(args.data_dir, "ckpt.pt"))
    print("Saved ckpt to", os.path.join(args.data_dir, "ckpt.pt"))

if __name__ == "__main__":
    main()
