import os
import json
import argparse
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
from losses import total_loss

NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit",
]
TARGET_COL = "delay"


def load_scalers(data_dir):
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")
    if not os.path.exists(ss_path) or not os.path.exists(ys_path):
        raise FileNotFoundError("Missing scaler_stats.json or y_scaler.json")
    stats = json.load(open(ss_path, "r"))
    yinfo = json.load(open(ys_path, "r"))
    x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    if abs(y_std) < 1e-12:
        y_std = 1.0
    return x_mean, x_std, y_mean, y_std, stats, yinfo


def _infer_hgat_hid(sd: dict) -> int:
    if sd is None:
        return 64
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    return 64


def _ensure_pol_bit(df):
    if "pol_bit" not in df.columns:
        if "pol" in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df["pol_bit"] = 0.0
    return df


def _ensure_numeric_cols(df):
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _norm_xy(df, x_mean, x_std, y_mean, y_std):
    df = _ensure_pol_bit(df.copy())
    df = _ensure_numeric_cols(df)
    x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
    x = (x_raw - x_mean) / x_std
    y_raw = df[TARGET_COL].astype(np.float32).values
    y = (y_raw - y_mean) / y_std
    cts = df["cell_type"].values
    return x, y, cts


class Stage1Dataset(Dataset):
    def __init__(self, df, x_mean, x_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        x, y, cts = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)
        self.x = x
        self.y = y
        self.cell_types = cts

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i]), self.cell_types[i]


class Stage2Dataset(Dataset):
    def __init__(self, csv_path, x_mean, x_std, y_mean, y_std):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        x, y, cts = _norm_xy(df, x_mean, x_std, y_mean, y_std)
        self.x, self.y, self.cell_types = x, y, cts

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i]), self.cell_types[i]


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


def precompute_z_from_src_map(data_dir, enc, device, src_map):
    """src_map: cell_type -> spice_file_path"""
    graph_cache = {}
    total_cells = len(src_map)
    start_time = time.time()
    for idx, (ct, path) in enumerate(src_map.items()):
        if idx % 10 == 0 or idx == total_cells - 1:
            elapsed = time.time() - start_time
            print(f"  -> Parsing {idx + 1}/{total_cells} cells... ({elapsed:.1f}s)", end='\r')

        full_path = path if os.path.exists(path) else os.path.join(data_dir, path)
        if os.path.exists(full_path):
            txt = open(full_path, 'r', encoding="utf-8", errors="ignore").read()
            devs = parse_transistors_spice(txt)
            _, pins = parse_top_subckt_pins(txt)
            if devs:
                g, feats, _ = build_dgl_graph_from_devs(devs, pins)
                graph_cache[ct] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    print(f"\n[Info] Cached {len(graph_cache)} source graphs.")
    return graph_cache


def precompute_z_from_tgt_spice(data_dir, tgt_spice, enc, device):
    """use meta.json key: tgt_subckt_by_cell"""
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    mapping = meta.get("tgt_subckt_by_cell", {})
    if not mapping:
        print("[Warn] meta.json has no tgt_subckt_by_cell")
        return {}

    if not os.path.exists(tgt_spice):
        cand = os.path.join(data_dir, tgt_spice)
        if os.path.exists(cand):
            tgt_spice = cand
        else:
            raise FileNotFoundError(f"Target SPICE not found: {tgt_spice}")

    print(f"[Info] Parsing Target SPICE: {tgt_spice}")
    sp_text = open(tgt_spice, "r", encoding="utf-8", errors="ignore").read()

    z_dict = {}
    enc.eval()
    with torch.no_grad():
        for ctype, sub_name in mapping.items():
            sub_txt = extract_subckt_text(sp_text, sub_name)
            if not sub_txt:
                continue
            devs = parse_transistors_spice(sub_txt)
            _, pins = parse_top_subckt_pins(sub_txt)
            if not devs:
                continue
            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[ctype] = z
    print(f"[Info] Pre-computed target Z for {len(z_dict)} cells.")
    return z_dict


@torch.no_grad()
def eval_mae_on_dataset(ds: Dataset, dl: DataLoader, enc, model, z_provider, y_mean, y_std, device, design_dim):
    """
    z_provider: callable(ct)-> z tensor [1, design_dim] or None
    returns MAE in ps
    """
    enc.eval()
    model.eval()
    err_sum = 0.0
    n = 0
    for xb, yb, cts in dl:
        xb = xb.to(device)
        yb = yb.to(device)

        z_list = []
        for ct in cts:
            z = z_provider(ct)
            if z is None:
                z_list.append(torch.zeros(1, design_dim, device=device))
            else:
                z_list.append(z)
        zb = torch.cat(z_list, dim=0)

        mu, _, _, _ = model(xb, zb)
        max_abs = 10.0
        mu_t = max_abs * torch.tanh(mu / max_abs)

        pred_ps = mu_t * y_std + y_mean
        true_ps = yb * y_std + y_mean
        err_sum += torch.abs(pred_ps - true_ps).sum().item()
        n += len(xb)
    return err_sum / max(1, n)


def split_src_by_celltype(df, val_ratio=0.1, seed=42):
    """æ cell_type åç»ååï¼é¿ååä¸ cell_type åæ¶åºç°å¨ train/val"""
    if val_ratio <= 0:
        return df, None

    rng = np.random.RandomState(seed)
    cell_types = df["cell_type"].astype(str).unique().tolist()
    rng.shuffle(cell_types)
    n_val = max(1, int(len(cell_types) * val_ratio))
    val_ct = set(cell_types[:n_val])

    is_val = df["cell_type"].astype(str).isin(val_ct)
    df_val = df[is_val].reset_index(drop=True)
    df_tr = df[~is_val].reset_index(drop=True)
    return df_tr, df_val


def run_stage1_pretraining(args, device, x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler):
    print("\n" + "=" * 60)
    print(" >>> STAGE 1: Source Domain Pre-training <<<")
    print("=" * 60)

    src_csv = os.path.join(args.data_dir, "src_delay.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Missing: {src_csv}")

    df_src = pd.read_csv(src_csv)
    if "cell_type" not in df_src.columns:
        raise RuntimeError("src_delay.csv must contain column: cell_type")

    df_tr, df_val = split_src_by_celltype(df_src, val_ratio=args.src_val_ratio, seed=args.seed)
    if df_val is not None:
        print(f"[S1] Split src by cell_type: train_rows={len(df_tr)} val_rows={len(df_val)} (val_ratio={args.src_val_ratio})")
    else:
        print(f"[S1] No src val split (src_val_ratio=0). Use train loss as best criterion. train_rows={len(df_tr)}")

    train_ds = Stage1Dataset(df_tr, x_mean, x_std, y_mean, y_std)
    val_ds = Stage1Dataset(df_val, x_mean, x_std, y_mean, y_std) if df_val is not None else None

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return torch.stack(xs), torch.stack(ys), cts

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=my_collate) if val_ds else None

    in_map = {"NET": 4, "PMOS": 2, "NMOS": 2}
    enc = HGATDesignEncoder(in_dim_map=in_map, hid=args.hid, out=args.design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=args.design_dim).to(device)

    optimizer = optim.Adam(list(enc.parameters()) + list(model.parameters()), lr=args.lr)

    scheduler = None
    if args.auto_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
        )

    # cache graphs for src cells
    with open(os.path.join(args.data_dir, "meta.json"), 'r') as f:
        src_map = json.load(f).get("src_spi_by_cell", {})
    print("[S1] Caching Source Graphs ...")
    graph_cache = precompute_z_from_src_map(args.data_dir, enc, device, src_map)

    def z_provider(ct):
        if ct in graph_cache:
            g, feats = graph_cache[ct]
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return z
        return None

    best_metric = float("inf")
    bad_epochs = 0

    best_path = os.path.join(args.save_dir, "ckpt_stage1_best.pt")
    last_path = os.path.join(args.save_dir, "ckpt_stage1_last.pt")

    for epoch in range(args.s1_epochs):
        enc.train()
        model.train()
        epoch_loss_val = 0.0
        count = 0

        for xb, yb, cts in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # build z batch
            z_list = []
            valid = []
            for i, ct in enumerate(cts):
                if ct in graph_cache:
                    g, feats = graph_cache[ct]
                    z = enc(g, feats)
                    if z.dim() == 1:
                        z = z.unsqueeze(0)
                    z_list.append(z)
                    valid.append(i)
            if not z_list:
                continue

            zb = torch.cat(z_list, dim=0)
            xb = xb[valid]
            yb = yb[valid]

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=args.s1_kl)

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(model.parameters()), args.grad_clip)
            optimizer.step()

            epoch_loss_val += loss.item() * len(yb)
            count += len(yb)

        avg_loss = epoch_loss_val / (count + 1e-6)

        # metric for "best"
        if val_dl is not None:
            src_val_mae = eval_mae_on_dataset(val_ds, val_dl, enc, model, z_provider, y_mean, y_std, device, args.design_dim)
            metric = src_val_mae
        else:
            src_val_mae = None
            metric = avg_loss

        if scheduler is not None:
            scheduler.step(metric)

        cur_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0:
            if src_val_mae is None:
                print(f"  [S1] Ep {epoch+1:3d}/{args.s1_epochs} | TrainLoss {avg_loss:.4f} | LR {cur_lr:.2e}")
            else:
                print(f"  [S1] Ep {epoch+1:3d}/{args.s1_epochs} | TrainLoss {avg_loss:.4f} | SrcValMAE {src_val_mae:.4f} ps | LR {cur_lr:.2e}")

        # save last every epoch end
        torch.save({
            "model": model.state_dict(),
            "enc": enc.state_dict(),
            "hgat_in_dim_map": in_map,
            "design_dim": args.design_dim,
            "scaler_stats": scaler_stats,
            "y_scaler": y_scaler,
            "stage": "stage1_last",
            "epoch": epoch + 1,
        }, last_path)

        # save best
        if metric < best_metric:
            best_metric = metric
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_map,
                "design_dim": args.design_dim,
                "scaler_stats": scaler_stats,
                "y_scaler": y_scaler,
                "stage": "stage1_best",
                "epoch": epoch + 1,
                "best_metric": best_metric,
                "criterion": "src_val_mae" if val_dl is not None else "train_loss",
            }, best_path)
        else:
            bad_epochs += 1
            if args.early_patience and args.early_patience > 0 and bad_epochs >= args.early_patience:
                print(f"  [S1] Early stop at ep={epoch+1} (no improve for {bad_epochs} epochs).")
                break

    print(f"[S1] Best saved: {best_path}")
    print(f"[S1] Last saved: {last_path}")
    return best_path


def run_stage2_transfer(args, device, src_ckpt_path, x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler):
    print("\n" + "=" * 60)
    print(" >>> STAGE 2: Target Fine-tuning (Transfer) <<<")
    print("=" * 60)

    if not os.path.exists(src_ckpt_path):
        raise FileNotFoundError(f"Missing src ckpt: {src_ckpt_path}")

    print(f"[S2] Loading Stage1 ckpt: {src_ckpt_path}")
    state = torch.load(src_ckpt_path, map_location=device)

    design_dim = int(state.get("design_dim", 64))
    enc_hid = _infer_hgat_hid(state["enc"])
    in_map = state.get("hgat_in_dim_map", {"NET": 4, "PMOS": 2, "NMOS": 2})

    enc = HGATDesignEncoder(in_dim_map=in_map, hid=enc_hid, out=design_dim).to(device)
    enc.load_state_dict(state["enc"], strict=True)

    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=args.hid, design_dim_override=design_dim).to(device)
    model.load_state_dict(state["model"], strict=False)

    # unfreeze encoder
    print("[S2] Fine-tuning HGAT encoder (Unfrozen).")
    optimizer = optim.Adam([
        # 1. Regressor (MLP): Î¬³ÖÔ­ÓÐµÄÑ§Ï°ÂÊ²ßÂÔ (args.lr * 0.5)
        {'params': model.parameters(), 'lr': args.lr * 0.5},

        # 2. Encoder (HGAT): Ê¹ÓÃ·Ç³£Ð¡µÄÑ§Ï°ÂÊ (ÀýÈç 1% ~ 10% µÄÖ÷Ñ§Ï°ÂÊ)
        #    ÕâÑù¿ÉÒÔÎ¢µ÷ÌØÕ÷£¬¶ø²»ÊÇÆÆ»µÔ¤ÑµÁ·µÄ½á¹¹
        {'params': enc.parameters(), 'lr': args.lr * 0.01}
    ])
    scheduler = None
    if args.auto_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
        )

    # z_map for target cells
    z_map = precompute_z_from_tgt_spice(args.data_dir, args.tgt_spice, enc, device)

    train_csv = os.path.join(args.data_dir, "tgt_train.csv")
    val_csv = os.path.join(args.data_dir, "tgt_val.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError("Missing tgt_train.csv")

    train_ds = Stage2Dataset(train_csv, x_mean, x_std, y_mean, y_std)
    val_ds = Stage2Dataset(val_csv, x_mean, x_std, y_mean, y_std) if os.path.exists(val_csv) else None

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return torch.stack(xs), torch.stack(ys), cts

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=my_collate) if val_ds else None

    print(f"[S2] Train rows={len(train_ds)} | Val rows={len(val_ds) if val_ds else 0}")

    def z_provider(ct):
        return z_map.get(ct, None)

    best_mae = float("inf")
    bad_epochs = 0

    best_path = os.path.join(args.save_dir, "ckpt_transfer_best.pt")
    last_path = os.path.join(args.save_dir, "ckpt_transfer_last.pt")

    for epoch in range(args.s2_epochs):
        model.train()
        enc.train()
        epoch_loss = 0.0

        for xb, yb, cts in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            z_list = []
            for ct in cts:
                z = z_map.get(ct, None)
                if z is None:
                    z_list.append(torch.zeros(1, design_dim, device=device))
                else:
                    z_list.append(z)
            zb = torch.cat(z_list, dim=0)

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=args.s2_kl)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        avg_train_loss = epoch_loss / max(1, len(train_ds))

        val_mae = None
        if val_dl is not None:
            val_mae = eval_mae_on_dataset(val_ds, val_dl, enc, model, z_provider, y_mean, y_std, device, design_dim)
            metric = val_mae
        else:
            metric = avg_train_loss

        if scheduler is not None:
            scheduler.step(metric)

        cur_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 10 == 0:
            if val_mae is None:
                print(f"  [S2] Ep {epoch+1:3d}/{args.s2_epochs} | TrainLoss {avg_train_loss:.4f} | LR {cur_lr:.2e}")
            else:
                print(f"  [S2] Ep {epoch+1:3d}/{args.s2_epochs} | TrainLoss {avg_train_loss:.4f} | ValMAE {val_mae:.4f} ps | LR {cur_lr:.2e}")

        # save last
        torch.save({
            "model": model.state_dict(),
            "enc": enc.state_dict(),  # frozen but saved for completeness
            "hgat_in_dim_map": in_map,
            "design_dim": design_dim,
            "scaler_stats": scaler_stats,
            "y_scaler": y_scaler,
            "stage": "stage2_last",
            "epoch": epoch + 1,
        }, last_path)

        # save best (by val_mae if present)
        if val_mae is not None and val_mae < best_mae:
            best_mae = val_mae
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_map,
                "design_dim": design_dim,
                "scaler_stats": scaler_stats,
                "y_scaler": y_scaler,
                "stage": "stage2_best",
                "epoch": epoch + 1,
                "best_val_mae": best_mae,
            }, best_path)
        elif val_mae is not None:
            bad_epochs += 1
            if args.early_patience and args.early_patience > 0 and bad_epochs >= args.early_patience:
                print(f"  [S2] Early stop at ep={epoch+1} (no improve for {bad_epochs} epochs).")
                break

    print(f"[S2] Best saved: {best_path} (best_val_mae={best_mae:.4f} ps)")
    print(f"[S2] Last saved: {last_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./output")
    parser.add_argument("--tgt_spice", default="asap7.sp")

    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "transfer", "all"])
    parser.add_argument("--src_ckpt", type=str, default="")

    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--design_dim", type=int, default=64)

    parser.add_argument("--s1_epochs", type=int, default=50)
    parser.add_argument("--s2_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--auto_lr", action="store_true")
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # âæ°å¢ï¼Stage1/Stage2 KL æéå¯è°
    parser.add_argument("--s1_kl", type=float, default=0.05)
    parser.add_argument("--s2_kl", type=float, default=0.01)

    # âæ°å¢ï¼æºå val ååï¼æ¨è 0.1ï¼
    parser.add_argument("--src_val_ratio", type=float, default=0.0,
                        help="Split src_delay by cell_type for validation. 0 means disable.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("[Info] Loading scalers...")
    x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler = load_scalers(args.data_dir)

    current_ckpt = args.src_ckpt

    if args.mode in ["pretrain", "all"]:
        current_ckpt = run_stage1_pretraining(
            args, device, x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler
        )

    if args.mode in ["transfer", "all"]:
        if not current_ckpt or not os.path.exists(current_ckpt):
            print("[Error] Missing stage1 ckpt for stage2.")
            if args.mode == "transfer":
                print("Please pass --src_ckpt")
            return

        run_stage2_transfer(
            args, device, current_ckpt, x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler
        )


if __name__ == "__main__":
    main()
