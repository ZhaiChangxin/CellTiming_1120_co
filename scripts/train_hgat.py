# === Python代码文件: train_hgat.py ===
import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler

# === 依赖导入 ===
from model import DisentangledRegressor, DomainClassifier
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
from losses import total_loss

# ====== 全局配置 ======
NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp", "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv", "stack_pu", "stack_pd", "log_slew", "log_cap",
    "req_p", "req_n", "rc_p", "rc_n", "rc_eff", "req_eff",
    "inv_v", "inv_temp", "pn_balance", "pol_bit",
]
TARGET_COL = "delay"


# ==========================================
#      Data Utils
# ==========================================
def load_scalers(data_dir):
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")
    if not os.path.exists(ss_path): raise FileNotFoundError("Missing scaler_stats.json")

    stats = json.load(open(ss_path, "r"))
    yinfo = json.load(open(ys_path, "r"))
    x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    return x_mean, x_std, y_mean, y_std


class BaseDataset(Dataset):
    def __init__(self, csv_path, x_mean, x_std, y_mean, y_std):
        self.df = pd.read_csv(csv_path)
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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i]), self.cell_types[i]


def common_collate(batch):
    xs, ys, cts = zip(*batch)
    return torch.stack(xs), torch.stack(ys), cts


def precompute_source_graphs(data_dir, device):
    """预加载源域图数据"""
    print(" >>> Pre-computing Source Graphs...")
    graph_cache = {}
    with open(os.path.join(data_dir, "meta.json"), 'r') as f:
        src_map = json.load(f).get("src_spi_by_cell", {})

    for ct, path in src_map.items():
        full_path = path if os.path.exists(path) else os.path.join(data_dir, path)
        if os.path.exists(full_path):
            txt = open(full_path, 'r').read()
            devs = parse_transistors_spice(txt)
            _, pins = parse_top_subckt_pins(txt)
            if devs:
                g, feats, _ = build_dgl_graph_from_devs(devs, pins)
                graph_cache[ct] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    return graph_cache


def precompute_target_embeddings(data_dir, spice_file, enc, device):
    """预计算目标域 Embedding (冻结 HGAT 后使用)"""
    print(" >>> Pre-computing Target Embeddings...")
    z_dict = {}
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r"))
    mapping = meta.get("tgt_subckt_by_cell", {})

    sp_path = os.path.join(data_dir, spice_file)
    if not os.path.exists(sp_path): return {}

    import re
    sp_text = open(sp_path, "r").read()

    def get_subckt(name):
        patt = re.compile(r"\s*\.subckt\s+%s\b(.*?)\.ends\b" % re.escape(name), re.DOTALL | re.IGNORECASE)
        m = patt.search(sp_text)
        return m.group(0) if m else ""

    enc.eval()
    with torch.no_grad():
        for ctype, sub_name in mapping.items():
            txt = get_subckt(sub_name)
            if not txt: continue
            devs = parse_transistors_spice(txt)
            _, pins = parse_top_subckt_pins(txt)
            if devs:
                g, feats, _ = build_dgl_graph_from_devs(devs, pins)
                z = enc(g.to(device), {k: v.to(device) for k, v in feats.items()})
                if z.dim() == 1: z = z.unsqueeze(0)
                z_dict[ctype] = z
    return z_dict


# ==========================================
#      PHASE 1: 初始化稳定特征提取器 (优化版: Scheduler + Clipping + Save Best)
# ==========================================
def run_phase1(args, device, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 60)
    print(" [Phase 1] Initialize Stable Feature Extractor (Source Only)")
    print(" [Config] Added Scheduler, Grad Clip & Save Best")
    print("=" * 60)

    # 1. Data
    ds = BaseDataset(os.path.join(args.data_dir, "src_delay.csv"), x_mean, x_std, y_mean, y_std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=common_collate)
    graph_cache = precompute_source_graphs(args.data_dir, device)

    # 2. Model
    in_map = {"NET": 4, "PMOS": 2, "NMOS": 2}
    enc = HGATDesignEncoder(in_dim_map=in_map, hid=args.hid, out=args.design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=args.design_dim).to(
        device)

    optimizer = optim.Adam(list(enc.parameters()) + list(model.parameters()), lr=args.lr)

    # [新增] 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5
    )

    # 3. Training
    enc.train();
    model.train()
    best_loss = float('inf')
    save_path = os.path.join(args.save_dir, "ckpt_phase1.pt")

    for epoch in range(args.epochs_p1):
        epoch_loss = 0
        count = 0
        steps = 0

        for xs, ys, cts in loader:
            xs, ys = xs.to(device), ys.to(device)

            z_list = []
            valid_idx = []
            for i, ct in enumerate(cts):
                if ct in graph_cache:
                    g, feats = graph_cache[ct]
                    z_list.append(enc(g, feats).view(1, -1))
                    valid_idx.append(i)

            if not z_list: continue
            zb = torch.cat(z_list, dim=0)
            xb, yb = xs[valid_idx], ys[valid_idx]

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)

            # Loss
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=0.01)

            loss.backward()

            # [新增] 梯度裁剪 (防止 Loss 震荡)
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(model.parameters()), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item() * len(yb)
            count += len(yb)
            steps += 1

        avg_loss = epoch_loss / count if count > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']

        # [新增] 更新 Scheduler
        scheduler.step(avg_loss)

        print(f"  [P1] Epoch {epoch + 1}/{args.epochs_p1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # [新增] 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "enc": enc.state_dict(),
                "model": model.state_dict(),
                "config": {"design_dim": args.design_dim, "hid": args.hid}
            }, save_path)

    print(f"  -> Phase 1 Complete. Best Loss: {best_loss:.4f}. Saved to {save_path}")
    return save_path


# ==========================================
#      PHASE 2: 学习领域不变特征 (DANN) (优化版: Focus on Task Loss stability)
# ==========================================
def run_phase2(args, device, p1_ckpt, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 60)
    print(" [Phase 2] Domain Adaptation (DANN) - Adversarial Training")
    print(" [Config] Stabilizing based on Task Loss")
    print("=" * 60)

    # 1. Load Phase 1
    # 兼容性加载
    try:
        ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
    except:
        ckpt = torch.load(p1_ckpt, map_location=device)

    enc = HGATDesignEncoder(in_dim_map={"NET": 4, "PMOS": 2, "NMOS": 2},
                            hid=args.hid, out=args.design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid,
                                  design_dim_override=args.design_dim).to(device)
    enc.load_state_dict(ckpt["enc"])
    model.load_state_dict(ckpt["model"])

    # 2. Domain Classifier
    domain_clf = DomainClassifier(in_dim=model.head_in_dim, hid=64).to(device)

    # 3. Data
    src_ds = BaseDataset(os.path.join(args.data_dir, "src_delay.csv"), x_mean, x_std, y_mean, y_std)
    tgt_ds = BaseDataset(os.path.join(args.data_dir, "tgt_train.csv"), x_mean, x_std, y_mean, y_std)

    min_len = min(len(src_ds), len(tgt_ds))
    loader_src = DataLoader(src_ds, batch_size=args.batch_size,
                            sampler=RandomSampler(src_ds, replacement=True, num_samples=min_len),
                            collate_fn=common_collate)
    loader_tgt = DataLoader(tgt_ds, batch_size=args.batch_size,
                            sampler=RandomSampler(tgt_ds, replacement=True, num_samples=min_len),
                            collate_fn=common_collate)

    src_graphs = precompute_source_graphs(args.data_dir, device)
    tgt_z_map = precompute_target_embeddings(args.data_dir, args.tgt_spice, enc, device)

    # 4. Freeze Regression Head
    for p in model.head.parameters(): p.requires_grad = False
    for p in model.mu.parameters(): p.requires_grad = False
    for p in model.log_var.parameters(): p.requires_grad = False

    # Optimizer & Scheduler
    # DANN 训练通常比较困难，可以尝试比 P1 稍小的初始学习率
    optimizer = optim.Adam(
        list(enc.parameters()) + list(model.enc.parameters()) +
        list(model.split_node.parameters()) + list(domain_clf.parameters()),
        lr=args.lr * 0.5
    )

    # [新增] 监控 Task Loss 来调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True, min_lr=1e-5
    )

    bce_loss = nn.BCEWithLogitsLoss()
    enc.train();
    model.train();
    domain_clf.train()

    best_task_loss = float('inf')
    save_path = os.path.join(args.save_dir, "ckpt_phase2.pt")

    for epoch in range(args.epochs_p2):
        total_d_loss = 0
        total_t_loss = 0
        steps = 0

        iter_tgt = iter(loader_tgt)

        for xs, ys, cts in loader_src:
            try:
                xt, _, ctt = next(iter_tgt)
            except StopIteration:
                break

            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            # Prepare Source Z
            zs_list = []
            valid_s = []
            for i, ct in enumerate(cts):
                if ct in src_graphs:
                    g, feats = src_graphs[ct]
                    zs_list.append(enc(g, feats).view(1, -1))
                    valid_s.append(i)
            if not zs_list: continue
            zs = torch.cat(zs_list, dim=0)
            xs_valid, ys_valid = xs[valid_s], ys[valid_s]

            # Prepare Target Z
            zt_list = []
            valid_t = []
            for i, ct in enumerate(ctt):
                if ct in tgt_z_map:
                    zt_list.append(tgt_z_map[ct])
                    valid_t.append(i)
            if not zt_list: continue
            zt = torch.cat(zt_list, dim=0).detach()
            xt_valid = xt[valid_t]

            # --- Forward ---
            # GRL Alpha: 动态增加对抗强度
            p = float(steps + epoch * len(loader_src)) / (args.epochs_p2 * len(loader_src))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            feat_s = model(xs_valid, zs, return_feat=True)
            feat_t = model(xt_valid, zt, return_feat=True)

            # Task Loss (只看 Source)
            mu_s, logv_s, _, _ = model(xs_valid, zs)
            # 这里我们只取 NLL 作为 Task Loss 的指标
            task_loss_val = total_loss(ys_valid, mu_s, logv_s, None, None)[1]

            # Domain Loss
            d_pred_s = domain_clf(feat_s, alpha)
            d_pred_t = domain_clf(feat_t, alpha)
            label_s = torch.zeros(d_pred_s.size(0), 1).to(device)
            label_t = torch.ones(d_pred_t.size(0), 1).to(device)
            dom_loss = bce_loss(d_pred_s, label_s) + bce_loss(d_pred_t, label_t)

            loss = task_loss_val + dom_loss

            optimizer.zero_grad()
            loss.backward()

            # [新增] 梯度裁剪 (关键！对抗训练很容易梯度爆炸)
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(model.parameters()) + list(domain_clf.parameters()),
                max_norm=5.0
            )

            optimizer.step()

            total_t_loss += task_loss_val.item()
            total_d_loss += dom_loss.item()
            steps += 1

        avg_t_loss = total_t_loss / steps if steps > 0 else 0
        avg_d_loss = total_d_loss / steps if steps > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(avg_t_loss)

        print(
            f"  [P2] Epoch {epoch + 1} | Task Loss: {avg_t_loss:.4f} | Dom Loss: {avg_d_loss:.4f} | LR: {current_lr:.2e}")

        # [新增] 只有当 Task Loss (源域回归性能) 变好时才保存
        if avg_t_loss < best_task_loss:
            best_task_loss = avg_t_loss
            torch.save({
                "enc": enc.state_dict(),
                "model": model.state_dict(),
                "config": ckpt["config"]
            }, save_path)

    print(f"  -> Phase 2 Complete. Best Task Loss: {best_task_loss:.4f}. Saved to {save_path}")
    return save_path


# ==========================================
#      PHASE 3 (修复版: 去除 Dropout，保留退火)
# ==========================================
def run_phase3(args, device, p2_ckpt, x_mean, x_std, y_mean, y_std):
    print("\n" + "=" * 60)
    print(" [Phase 3] Uncertainty Adaptation (BNN Fine-tuning)")
    print(f" [Config] KL Max Scale: {args.kl_scale} | Strategy: Sigmoid Annealing | Dropout: OFF")
    print("=" * 60)

    # 1. 加载模型
    try:
        ckpt = torch.load(p2_ckpt, map_location=device, weights_only=False)
    except:
        ckpt = torch.load(p2_ckpt, map_location=device)

    enc = HGATDesignEncoder(in_dim_map={"NET": 4, "PMOS": 2, "NMOS": 2},
                            hid=args.hid, out=args.design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid,
                                  design_dim_override=args.design_dim).to(device)
    enc.load_state_dict(ckpt["enc"])
    model.load_state_dict(ckpt["model"])

    # 2. [核心修复] 将 Dropout 设为 0.0！BNN 不需要额外的 Dropout。
    model.convert_to_bnn(dropout_p=0.0)

    # 3. 数据准备
    ds = BaseDataset(os.path.join(args.data_dir, "tgt_train.csv"), x_mean, x_std, y_mean, y_std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=common_collate)
    tgt_z_map = precompute_target_embeddings(args.data_dir, args.tgt_spice, enc, device)

    # 冻结特征提取器
    for p in enc.parameters(): p.requires_grad = False

    # 优化器
    # [修改] 既然继承了权重，学习率就不要太大了，防止把好权重震坏了
    start_lr = args.lr * 0.1  # 推荐：0.001 * 0.1 = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    # [修改] Scheduler 也可以更灵敏一点
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    model.train()

    best_loss = float('inf')
    patience_limit = 20
    patience_counter = 0
    final_save_path = os.path.join(args.save_dir, "ckpt_phase3_final.pt")

    len_loader = len(loader)

    for epoch in range(args.epochs_p3):
        epoch_nll = 0
        epoch_kl = 0
        epoch_total = 0
        steps = 0

        for batch_idx, (xs, ys, cts) in enumerate(loader):
            # --- KL Annealing ---
            current_step = epoch * len_loader + batch_idx
            total_steps = args.epochs_p3 * len_loader
            p = float(current_step) / total_steps

            # Sigmoid Annealing
            beta = args.kl_scale * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            # --------------------

            xs, ys = xs.to(device), ys.to(device)
            z_list = [tgt_z_map[ct] if ct in tgt_z_map else torch.zeros(1, args.design_dim).to(device) for ct in cts]
            zb = torch.cat(z_list, dim=0).detach()

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xs, zb)

            # Loss
            nll = total_loss(ys, mu, logv, None, None)[1]
            kl_raw = model.get_bnn_kl_loss() / len(ds)
            kl_weighted = kl_raw * beta

            loss = nll + kl_weighted

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_nll += nll.item()
            epoch_kl += kl_weighted.item()
            epoch_total += loss.item()
            steps += 1

        avg_loss = epoch_total / steps
        avg_nll = epoch_nll / steps
        avg_kl = epoch_kl / steps

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)

        print(
            f"  [P3] Epoch {epoch + 1} | Loss: {avg_loss:.4f} (NLL: {avg_nll:.4f}, KL: {avg_kl:.4f}, Beta: {beta:.4f}) | LR: {current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "enc": enc.state_dict(),
                "model": model.state_dict(),
                "config": ckpt.get("config", {"hid": args.hid, "design_dim": args.design_dim})
            }, final_save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"  [Stop] Early stopping triggered.")
            break

    print(f"  -> Phase 3 Complete. Best Loss: {best_loss:.4f}. Saved to {final_save_path}")


# ==========================================
#      Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./output")
    parser.add_argument("--tgt_spice", default="asap7.sp")
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--design_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--epochs_p1", type=int, default=50)
    parser.add_argument("--epochs_p2", type=int, default=50)
    parser.add_argument("--epochs_p3", type=int, default=100)

    # [修复点] 必须添加这一行，否则 args.kl_scale 会报错
    parser.add_argument("--kl_scale", type=float, default=5.0, help="Scaling factor for BNN KL loss in Phase 3")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)

    # 顺序执行三阶段
    # 注意：如果你之前跑过了 P1 和 P2 且保存了模型，
    # 为了节省时间，你可以注释掉下面两行，直接指定 p2_ckpt 路径
    ckpt_p1 = run_phase1(args, device, x_mean, x_std, y_mean, y_std)
    ckpt_p2 = run_phase2(args, device, ckpt_p1, x_mean, x_std, y_mean, y_std)

    # 如果你想跳过 P1/P2 直接调试 P3，可以使用如下写法（需确保文件存在）：
    # ckpt_p2 = os.path.join(args.save_dir, "ckpt_phase2.pt")

    run_phase3(args, device, ckpt_p2, x_mean, x_std, y_mean, y_std)


if __name__ == "__main__":
    main()

