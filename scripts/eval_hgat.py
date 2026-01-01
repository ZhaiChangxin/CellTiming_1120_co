import os
import json
import argparse
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------- ×Ô¶¨ÒåÄ£¿éµ¼Èë -----------------
from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins

# ====== ÌØÕ÷ÁÐ¶¨Òå (±£³ÖÓë train_hgat.py Ò»ÖÂ) ======
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


# ====== 1. Êý¾Ý¼ÓÔØÓëÔ¤´¦Àí¹¤¾ß ======
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

    return x_mean, x_std, y_mean, y_std


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


def _infer_hgat_hid_from_state(sd: dict) -> int:
    if sd is None:
        return 64
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    for k, v in sd.items():
        if k.startswith("readout.") and v.dim() == 2:
            return v.shape[1]
    return 64


def _reshape_main_net_to_ckpt(model, ckpt_model):
    import torch.nn as nn
    if "enc.0.weight" in ckpt_model:
        model.enc[0] = nn.Linear(ckpt_model["enc.0.weight"].shape[1], ckpt_model["enc.0.weight"].shape[0])
    if "enc.2.weight" in ckpt_model:
        model.enc[2] = nn.Linear(ckpt_model["enc.2.weight"].shape[1], ckpt_model["enc.2.weight"].shape[0])
    if "split_node.weight" in ckpt_model:
        model.split_node = nn.Linear(ckpt_model["split_node.weight"].shape[1], ckpt_model["split_node.weight"].shape[0])
    if "head.0.weight" in ckpt_model:
        model.head[0] = nn.Linear(ckpt_model["head.0.weight"].shape[1], ckpt_model["head.0.weight"].shape[0])
    if "head.2.weight" in ckpt_model:
        model.head[2] = nn.Linear(ckpt_model["head.2.weight"].shape[1], ckpt_model["head.2.weight"].shape[0])
    if "mu.weight" in ckpt_model:
        model.mu = nn.Linear(ckpt_model["mu.weight"].shape[1], 1)
    if "log_var.weight" in ckpt_model:
        model.log_var = nn.Linear(ckpt_model["log_var.weight"].shape[1], 1)
    return model


# ====== 2. Dataset ======
class EvalDataset(Dataset):
    def __init__(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)

        if "pol_bit" not in df.columns:
            if "pol" in df.columns:
                df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
            else:
                df["pol_bit"] = 0.0

        for c in NUMERIC_COLS:
            if c not in df.columns:
                df[c] = 0.0

        self.x = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        self.cell_types = df["cell_type"].values

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
    return xs, ys, cts


# ====== 3. Embedding Ô¤¼ÆËã ======
def prepare_target_embeddings(data_dir, tgt_spice_path, enc, device):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    tgt_map = meta.get("tgt_subckt_by_cell", {})
    if not tgt_map:
        return {}

    if not os.path.exists(tgt_spice_path):
        cand = meta.get("tgt_sp_file", "")
        if cand:
            p1 = os.path.join(data_dir, cand)
            if os.path.exists(p1):
                tgt_spice_path = p1
            elif os.path.exists(cand):
                tgt_spice_path = cand

    if not os.path.exists(tgt_spice_path):
        return {}

    print(f"[Info] Parsing Target SPICE: {tgt_spice_path}")
    sp_text = open(tgt_spice_path, "r", encoding="utf-8", errors="ignore").read()

    z_dict = {}
    enc.eval()
    with torch.no_grad():
        for cell_type, subckt_name in tgt_map.items():
            sub_txt = extract_subckt_text(sp_text, subckt_name)
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
            if z.dim() == 1: z = z.unsqueeze(0)
            z_dict[cell_type] = z
    return z_dict


def prepare_source_embeddings(data_dir, enc, device):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    src_map = meta.get("src_spi_by_cell", {})
    z_dict = {}
    enc.eval()

    print("[Info] Parsing Source SPICE files...")
    with torch.no_grad():
        for cell_type, rel_path in src_map.items():
            if os.path.exists(rel_path):
                sp_path = rel_path
            else:
                sp_path = os.path.join(data_dir, rel_path)

            if not os.path.exists(sp_path):
                continue

            with open(sp_path, "r", encoding="utf-8", errors="ignore") as f_sp:
                sp_text = f_sp.read()

            devs = parse_transistors_spice(sp_text)
            _, pins = parse_top_subckt_pins(sp_text)
            if not devs:
                continue

            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}
            z = enc(g, feats)
            if z.dim() == 1: z = z.unsqueeze(0)
            z_dict[cell_type] = z
    return z_dict


# ====== 4. Ö´ÐÐµ¥´ÎÆÀ¹À ======
def run_eval_single(csv_path, z_map, tag, model, x_mean_t, x_std_t, y_mean, y_std, device, design_dim):
    if csv_path is None or not os.path.exists(csv_path):
        return

    print(f"[Info] Evaluating {tag} on: {csv_path}")
    ds = EvalDataset(csv_path)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=eval_collate)

    preds, gts = [], []

    model.eval()
    with torch.no_grad():
        for xb, yb, cts in dl:
            xb = xb.to(device)
            xb = (xb - x_mean_t) / x_std_t

            z_list = []
            for ct in cts:
                if ct in z_map:
                    z_list.append(z_map[ct])
                else:
                    z_list.append(torch.zeros(1, design_dim, device=device))
            zb = torch.cat(z_list, dim=0)

            mu, _, _, _ = model(xb, zb)
            max_abs = 10.0
            mu = max_abs * torch.tanh(mu / max_abs)

            mu_np = mu.cpu().numpy()
            mu_ps = mu_np * y_std + y_mean
            preds.append(mu_ps)

            if yb is not None:
                gts.append(yb.numpy())

    y_pred = np.concatenate(preds, axis=0)

    if len(gts) > 0:
        y_true = np.concatenate(gts, axis=0)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print("-" * 50)
        print(f"{tag} Results ({len(y_pred)} samples):")
        print(f"  MAE : {mae:.5f} (ps)")
        print(f"  MSE : {mse:.5f} (ps^2)")
        print(f"  R2  : {r2:.4f}")
        print("-" * 50)

        out_df = pd.DataFrame({"pred": y_pred.flatten(), "label": y_true.flatten()})
    else:
        out_df = pd.DataFrame({"pred": y_pred.flatten()})

    out_name = f"eval_result_{tag.lower()}.csv"
    out_path = os.path.join(os.path.dirname(csv_path), out_name)
    out_df.to_csv(out_path, index=False)


# ====== 5. Checkpoint ÆÀ¹À Session ======
def evaluate_checkpoint_session(
        ckpt_path,
        tag_prefix,
        args,
        device,
        x_mean_t, x_std_t, y_mean, y_std,
        eval_flags,
        csv_paths
):
    if not eval_flags or not any(eval_flags.values()):
        return

    print(f"\n{'#' * 60}")
    print(f"[Session] Checkpoint: {ckpt_path}")
    print(f"[Session] Description: {tag_prefix if tag_prefix else 'Main Transfer Model'}")
    print(f"{'#' * 60}\n")

    try:
        from numpy.core.multiarray import _reconstruct
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([_reconstruct])
    except:
        pass

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

    ckpt_model = state.get("model", state)
    ckpt_enc = state.get("enc", None)
    design_dim = int(state.get("design_dim", args.design_dim))
    in_map = state.get("hgat_in_dim_map", {"NET": 4, "PMOS": 2, "NMOS": 2})

    enc_hid = _infer_hgat_hid_from_state(ckpt_enc)
    enc = HGATDesignEncoder(in_dim_map=in_map, hid=enc_hid, out=design_dim).to(device)
    if ckpt_enc:
        enc.load_state_dict(ckpt_enc, strict=True)

    model = DisentangledRegressor(
        in_dim=len(NUMERIC_COLS),
        hid=args.hid,
        design_dim_override=design_dim,
    )
    if "model" in state:
        model = _reshape_main_net_to_ckpt(model, ckpt_model)
        model.load_state_dict(ckpt_model, strict=True)

    model = model.to(device)
    model.eval()

    z_src = None
    z_tgt = None

    # Ö»ÔÚÐèÒªÊ±¼ÓÔØ Source Embedding
    if eval_flags.get('src', False):
        z_src = prepare_source_embeddings(args.data_dir, enc, device)

    # Ö»ÔÚÐèÒªÊ±¼ÓÔØ Target Embedding
    tgt_keys = ['tgt_train', 'tgt_val', 'tgt_test', 'tgt_custom']
    need_tgt = any([eval_flags.get(k, False) for k in tgt_keys])
    if need_tgt:
        z_tgt = prepare_target_embeddings(args.data_dir, args.tgt_spice, enc, device)

    tasks = [
        ('src', 'Source', z_src),
        ('tgt_train', 'Target_Train', z_tgt),
        ('tgt_val', 'Target_Val', z_tgt),
        ('tgt_test', 'Target_Test', z_tgt),
        ('tgt_custom', 'Target_Custom', z_tgt),
    ]

    for flag_key, suffix, z_map in tasks:
        if eval_flags.get(flag_key, False):
            run_eval_single(
                csv_path=csv_paths[flag_key],
                z_map=z_map,
                tag=f"{tag_prefix}{suffix}",
                model=model,
                x_mean_t=x_mean_t, x_std_t=x_std_t, y_mean=y_mean, y_std=y_std,
                device=device, design_dim=design_dim
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="Main Checkpoint (e.g. Transfer Best)")
    ap.add_argument("--src_ckpt", default="", help="Source Checkpoint (e.g. Pretrain Best)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--design_dim", type=int, default=64)
    ap.add_argument("--tgt_spice", type=str, default="asap7.sp")
    ap.add_argument("--csv", type=str, default="", help="Custom CSV")

    args = ap.parse_args()
    device = torch.device(args.device)

    print("[Info] Loading scalers...")
    try:
        x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)
    except Exception as e:
        print(f"[Error] {e}")
        return

    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)

    # 2. ¶¨ÒåÂ·¾¶Óë¿ÉÓÃÐÔ¼ì²é
    csv_paths = {
        'src': os.path.join(args.data_dir, "src_delay.csv"),
        'tgt_train': os.path.join(args.data_dir, "tgt_train.csv"),
        'tgt_val': os.path.join(args.data_dir, "tgt_val.csv"),
        'tgt_test': os.path.join(args.data_dir, "tgt_test.csv"),
        'tgt_custom': args.csv
    }

    # »ù´¡¿ÉÓÃµÄ flags
    available_flags = {k: False for k in csv_paths}
    if args.csv:
        available_flags['tgt_custom'] = True
    else:
        if os.path.exists(csv_paths['src']): available_flags['src'] = True
        if os.path.exists(csv_paths['tgt_train']): available_flags['tgt_train'] = True
        if os.path.exists(csv_paths['tgt_val']): available_flags['tgt_val'] = True
        if os.path.exists(csv_paths['tgt_test']): available_flags['tgt_test'] = True

    # 3. ¡¾ºËÐÄÂß¼­ÐÞ¸Ä¡¿ÆÀ¹À·¶Î§·ÖÀë

    # ²ßÂÔ A: ¶ÔÓÚ Main Checkpoint (Í¨³£ÊÇ Transfer Ä£ÐÍ)£¬ÎÒÃÇÍ¨³£Ö»¹ØÐÄËüÔÚ Target Êý¾ÝÉÏµÄ±íÏÖ
    # ³ý·ÇÓÃ»§Ã»´« src_ckpt£¬»òÕßÇ¿ÖÆÏëÒª¿´ source ÉÏµÄ±íÏÖ(´Ë´¦ÎªÁË½â¾öÄãµÄÎÊÌâ£¬Ä¬ÈÏ¹Ø±Õ)
    flags_for_main = available_flags.copy()
    flags_for_main['src'] = False  # <--- ¹Ø¼ü£º½ûÖ¹ÓÃ Transfer Ä£ÐÍÅÜ Source Êý¾Ý

    # ²ßÂÔ B: ¶ÔÓÚ Source Checkpoint (Pretrain Ä£ÐÍ)£¬ÎÒÃÇÖ»¹ØÐÄËüÔÚ Source Êý¾ÝÉÏµÄ±íÏÖ
    flags_for_src = {k: False for k in available_flags}
    flags_for_src['src'] = available_flags['src']  # <--- ¹Ø¼ü£ºÖ»¿ªÆô Source Êý¾Ý

    # 4. Ö´ÐÐÆÀ¹À

    # (A) ÆÀ¹À Main Checkpoint (Target Domain)
    evaluate_checkpoint_session(
        ckpt_path=args.ckpt,
        tag_prefix="",  # Êä³öÀï»áÏÔÊ¾ Target_Train µÈ
        args=args,
        device=device,
        x_mean_t=x_mean_t, x_std_t=x_std_t, y_mean=y_mean, y_std=y_std,
        eval_flags=flags_for_main,
        csv_paths=csv_paths
    )

    # (B) ÆÀ¹À Source Checkpoint (Source Domain)
    if args.src_ckpt and os.path.exists(args.src_ckpt):
        evaluate_checkpoint_session(
            ckpt_path=args.src_ckpt,
            tag_prefix="SrcBase_",  # Êä³öÀï»áÏÔÊ¾ SrcBase_Source
            args=args,
            device=device,
            x_mean_t=x_mean_t, x_std_t=x_std_t, y_mean=y_mean, y_std=y_std,
            eval_flags=flags_for_src,
            csv_paths=csv_paths
        )
    else:
        if available_flags['src']:
            print("[Info] Source checkpoint not provided (--src_ckpt), skipping Source evaluation.")


if __name__ == "__main__":
    main()