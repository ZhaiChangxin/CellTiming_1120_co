import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import DisentangledRegressor
from losses import GaussianNLL, cmd_loss, SupConLoss
from spi2graph import parse_transistors_spice, extract_wl_features, parse_top_subckt_pins
# 引入 hgat 相关的构建函数
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs

NUMERIC_COLS_BASE = ["slew", "cap", "voltage", "temp", "wp_over_wn", "wp_sum", "wn_sum", "is_inv", "stack_pu",
                     "stack_pd", "pol_bit"]
TARGET_COL = "delay"


# ... [Dataset 类代码保持不变，此处省略以节省篇幅，请保留原文件中的 CSVDataset 和 collate] ...
class CSVDataset(Dataset):
    def __init__(self, csv_path, tech_label, x_mean, x_std, y_mean, y_std):
        self.df = pd.read_csv(csv_path)
        # 修正：确保 pol_bit 逻辑兼容
        if "pol_bit" not in self.df.columns:
            if "pol" in self.df.columns:
                self.df["pol_bit"] = (self.df["pol"].astype(str) == "rise").astype(np.float32)
            else:
                self.df["pol_bit"] = 0.0

        self.df["is_inv"] = 1.0;
        self.df["stack_pu"] = 1.0;
        self.df["stack_pd"] = 1.0

        for c in NUMERIC_COLS_BASE:
            if c not in self.df.columns: self.df[c] = np.nan
        self.df[NUMERIC_COLS_BASE] = self.df[NUMERIC_COLS_BASE].fillna(0.0).astype(np.float32)

        self.x = self.df[NUMERIC_COLS_BASE].values.astype(np.float32)
        self.y = self.df[TARGET_COL].values.astype(np.float32)

        # scalers
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = np.float32(y_mean)
        self.y_std = np.float32(y_std)

        self.labels = np.full((len(self.df),), tech_label, dtype=np.int64)

    def __len__(self):
        return len(self.df)

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
    ap.add_argument("--lr", type=float, default=1e-3)  # 建议从 1e-3 开始
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--cmd_k", type=int, default=3)  # K=3 通常足够
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

    df_src_full = pd.read_csv(src_csv)
    df_tgt_l_full = pd.read_csv(tgt_l_csv)

    # --- 数据预处理与Scaler计算 ---
    def ensure_cols(df):
        if "pol_bit" not in df.columns:
            if "pol" in df.columns:
                df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
            else:
                df["pol_bit"] = 0.0
        for c in NUMERIC_COLS_BASE:
            if c not in df.columns: df[c] = 0.0
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

    # 保存 scaler 供 eval 使用
    scaler_stats = {
        "mean": {c: float(m) for c, m in zip(NUMERIC_COLS_BASE, x_mean)},
        "std": {c: float(s) for c, s in zip(NUMERIC_COLS_BASE, x_std)}
    }
    json.dump(scaler_stats, open(os.path.join(args.data_dir, "scaler_stats.json"), "w"), indent=2)
    json.dump({"mean": float(y_mean), "std": float(y_std)}, open(os.path.join(args.data_dir, "y_scaler.json"), "w"),
              indent=2)

    d_src = CSVDataset(src_csv, tech_label=0, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    d_tgt_l = CSVDataset(tgt_l_csv, tech_label=1, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    d_tgt_u = CSVDataset(tgt_u_csv, tech_label=1, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    loader_src = DataLoader(d_src, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)
    loader_tgt_l = DataLoader(d_tgt_l, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=collate)
    loader_tgt_u = DataLoader(d_tgt_u, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)

    # --- 构建 HGAT ---
    enc = None
    bundle_src = None
    bundle_tgt = None
    in_map = None
    design_dim_override = None

    if args.use_hgat:
        print("[Info] Loading SPICE for HGAT...")

        # 辅助函数：加载文本并解析
        def prepare_bundle(path):
            if not path or not os.path.exists(path): return None
            text = open(path, "r", encoding="utf-8", errors="ignore").read()
            devs = parse_transistors_spice(text)
            name, pins = parse_top_subckt_pins(text)
            return build_dgl_graph_from_devs(devs, pins)

        bundle_src = prepare_bundle(args.src_spice)
        bundle_tgt = prepare_bundle(args.tgt_spice)

        if bundle_src or bundle_tgt:
            # 使用存在的那个图的维度信息来初始化 encoder
            ref_bundle = bundle_src if bundle_src else bundle_tgt
            in_map = ref_bundle[2]
            # 初始化 Encoder
            enc = HGATDesignEncoder(in_map, hid=64, out=64, num_heads=1).to(args.device)
            design_dim_override = 64
            print("[Info] HGAT Encoder initialized.")
        else:
            print("[Warning] use_hgat is True but no valid SPICE found.")

    # --- 模型初始化 ---
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS_BASE), hid=args.hid,
                                  design_dim_override=design_dim_override).to(args.device)

    # === 关键修正 1: 将 enc 的参数加入优化器 ===
    params = list(model.parameters())
    if enc is not None:
        params += list(enc.parameters())

    opt = torch.optim.Adam(params, lr=args.lr)

    g_nll = GaussianNLL()
    supcon = SupConLoss()

    # 辅助：将图数据移到 GPU
    def move_bundle(bundle, device):
        if bundle is None: return None
        g, feats, _ = bundle
        g = g.to(device)
        feats = {k: v.to(device) for k, v in feats.items()}
        return g, feats

    b_src_gpu = move_bundle(bundle_src, args.device)
    b_tgt_gpu = move_bundle(bundle_tgt, args.device)

    # --- 训练循环 ---
    print(f"Start training for {args.epochs} epochs...")
    best_nll = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        if enc: enc.train()

        iters = max(len(loader_src), len(loader_tgt_u))
        avg = {"loss": 0, "nll": 0, "cmd": 0, "supcon": 0}

        src_it = iter(loader_src)
        tgt_l_it = iter(loader_tgt_l)
        tgt_u_it = iter(loader_tgt_u)

        for _ in range(iters):
            # 获取数据 Batch
            try:
                xs, ys, ls = next(src_it)
            except StopIteration:
                src_it = iter(loader_src); xs, ys, ls = next(src_it)

            try:
                xtl, ytl, ltl = next(tgt_l_it)
            except StopIteration:
                tgt_l_it = iter(loader_tgt_l); xtl, ytl, ltl = next(tgt_l_it)

            try:
                xtu, ytu, ltu = next(tgt_u_it)
            except StopIteration:
                tgt_u_it = iter(loader_tgt_u); xtu, ytu, ltu = next(tgt_u_it)

            xs, ys, ls = xs.to(args.device), ys.to(args.device), ls.to(args.device)
            xtl, ytl, ltl = xtl.to(args.device), ytl.to(args.device), ltl.to(args.device)
            xtu, ytu, ltu = xtu.to(args.device), ytu.to(args.device), ltu.to(args.device)

            opt.zero_grad()

            # === 关键修正 2: 在循环内部计算 Z，且保留梯度 (无 no_grad/detach) ===
            # 虽然图结构是固定的，但 enc 的参数在变，所以每次前向传播 z 都是新的计算图节点
            z_src_emb = None
            z_tgt_emb = None

            if enc is not None:
                if b_src_gpu: z_src_emb = enc(b_src_gpu[0], b_src_gpu[1])
                if b_tgt_gpu: z_tgt_emb = enc(b_tgt_gpu[0], b_tgt_gpu[1])

            # 模型前向
            # 注意：如果对应 spice 缺失，z 为 None，代码会报错。假设都有 spice。
            # zd_s, zd_tl 等是模型返回的 design 向量 (这里直接等于输入的 z_src_emb/z_tgt_emb)
            mu_s, logv_s, zn_s, zd_s = model(xs, z_src_emb)
            mu_tl, logv_tl, zn_tl, zd_tl = model(xtl, z_tgt_emb)

            # Unlabeled Target 用于 CMD 对齐
            _, _, _, zd_u = model(xtu, z_tgt_emb)

            # Loss 计算
            nll = g_nll(mu_s, logv_s, ys) + g_nll(mu_tl, logv_tl, ytl)

            # CMD Loss: 对齐 Source Design 和 Target Design (假设目标是拉近两者距离)
            # 注意：zd_s 和 zd_u 在这里本质上就是 z_src_emb 和 z_tgt_emb 的广播
            cmd = torch.tensor(0.0, device=args.device)
            if args.lambda_cmd > 0 and zd_s is not None and zd_u is not None:
                cmd = cmd_loss(zd_s, zd_u, K=args.cmd_k)

            # SupCon Loss: 对齐同类 Content (zn)
            s_con = torch.tensor(0.0, device=args.device)
            if args.lambda_supcon > 0:
                zn = torch.cat([zn_s, zn_tl], dim=0)
                labs = torch.cat([ls, ltl], dim=0)
                s_con = supcon(zn, labs)

            loss = nll + args.lambda_cmd * cmd + args.lambda_supcon * s_con

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if enc is not None:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
            opt.step()

            avg["loss"] += loss.item();
            avg["nll"] += nll.item()
            avg["cmd"] += cmd.item();
            avg["supcon"] += s_con.item()

        for k in avg: avg[k] /= iters
        print(
            f"[Epoch {epoch}] loss={avg['loss']:.4f} nll={avg['nll']:.4f} cmd={avg['cmd']:.4f} supcon={avg['supcon']:.4f}")

        # 保存最优模型
        if avg["nll"] < best_nll:
            best_nll = avg["nll"]
            best_state = {
                "model": model.state_dict(),
                "enc": (enc.state_dict() if enc is not None else None),
                "hgat_in_dim_map": in_map,
                "design_dim": (design_dim_override if design_dim_override is not None else 0),
            }
            torch.save(best_state, os.path.join(args.data_dir, "ckpt.pt"))

    print("Saved best ckpt to", os.path.join(args.data_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()