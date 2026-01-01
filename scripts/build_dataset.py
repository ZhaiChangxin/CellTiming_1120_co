# === Python´úÂëÎÄ¼þ: build_dataset.py (ÒÑÐÞ¸´ Scaler Éú³É) ===

import argparse
import os
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# ¼ÙÉèÕâÐ©¿âÎÄ¼þºÍÄã±¾µØ»·¾³Ò»ÖÂ
from parse_lib import parse_cell_arcs
from spi2graph import parse_transistors_spice, extract_wl_features

# ======================================================
# ÅäÖÃ£ºÒª±£ÁôµÄ cell ÀàÐÍ
# ======================================================

TARGET_CELL_TYPES = [
    "AND2X2", "AND2X4", "AND3X1", "AND3X2", "AND3X4", "AND4X1", "AND4X2",
    "BUFX2", "BUFX4", "BUFX8", "BUFX16",
    "INVX1", "INVX2", "INVX4", "INVX8",
    "NAND2X1", "NAND2X2", "NAND3X1", "NAND3X2",
    "NOR2X1", "NOR2X2", "NOR3X1", "NOR3X2",
    "OR2X2", "OR2X4", "OR3X1", "OR3X2", "OR3X4", "OR4X1", "OR4X2",
    "XNOR2X2", "XOR2X2",
]

# ±ØÐëÓë train_hgat.py ÖÐµÄÁÐ±í±£³ÖÒ»ÖÂ£¬ÓÃÓÚÉú³É scaler_stats.json
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

# -------- Ô´Óò£¨Nangate£©Ã¿¸ö cell ¶ÔÓ¦Ò»¸ö SPI ÎÄ¼þ --------
SRC_CELL_SPI_FILES = {
    "AND2X2": "AND2_X2_lpe.spi",
    "AND2X4": "AND2_X4_lpe.spi",
    "AND3X1": "AND3_X1_lpe.spi",
    "AND3X2": "AND3_X2_lpe.spi",
    "AND3X4": "AND2_X4_lpe.spi",
    "AND4X1": "AND4_X1_lpe.spi",
    "AND4X2": "AND4_X2_lpe.spi",
    "BUFX2": "BUF_X2_lpe.spi",
    "BUFX4": "BUF_X4_lpe.spi",
    "BUFX8": "BUF_X8_lpe.spi",
    "BUFX16": "BUF_X16_lpe.spi",
    "INVX1": "INV_X1_lpe.spi",
    "INVX2": "INV_X2_lpe.spi",
    "INVX4": "INV_X4_lpe.spi",
    "INVX8": "INV_X8_lpe.spi",
    "NAND2X1": "NAND2_X1_lpe.spi",
    "NAND2X2": "NAND2_X2_lpe.spi",
    "NAND3X1": "NAND3_X1_lpe.spi",
    "NAND3X2": "NAND3_X2_lpe.spi",
    "NOR2X1": "NOR2_X1_lpe.spi",
    "NOR2X2": "NOR2_X2_lpe.spi",
    "NOR3X1": "NOR3_X1_lpe.spi",
    "NOR3X2": "NOR3_X2_lpe.spi",
    "OR2X2": "OR2_X2_lpe.spi",
    "OR2X4": "OR2_X4_lpe.spi",
    "OR3X1": "OR3_X1_lpe.spi",
    "OR3X2": "OR3_X2_lpe.spi",
    "OR3X4": "OR3_X4_lpe.spi",
    "OR4X1": "OR4_X1_lpe.spi",
    "OR4X2": "OR4_X2_lpe.spi",
    "XOR2X2": "XOR2_X2_lpe.spi",
    "XNOR2X2": "XNOR2_X2_lpe.spi",
}

# -------- Ä¿±êÓò£¨ASAP7£©´ó SP ÎÄ¼þÀïµÄ subckt Ãû --------
ASAP7_CELL_SUBCKT = {
    "AND2X2": "AND2x2_ASAP7_6t_L",
    "AND2X4": "AND2x4_ASAP7_6t_L",
    "AND3X1": "AND3x1_ASAP7_6t_L",
    "AND3X2": "AND3x2_ASAP7_6t_L",
    "AND3X4": "AND3x4_ASAP7_6t_L",
    "AND4X1": "AND4x1_ASAP7_6t_L",
    "AND4X2": "AND4x2_ASAP7_6t_L",
    "BUFX2": "BUFx2_ASAP7_6t_L",
    "BUFX4": "BUFx4_ASAP7_6t_L",
    "BUFX8": "BUFx8_ASAP7_6t_L",
    "BUFX16": "BUFx16q_ASAP7_6t_L",
    "INVX1": "INVx1_ASAP7_6t_L",
    "INVX2": "INVx2_ASAP7_6t_L",
    "INVX4": "INVx4_ASAP7_6t_L",
    "INVX8": "INVx8_ASAP7_6t_L",
    "NAND2X1": "NAND2x1_ASAP7_6t_L",
    "NAND2X2": "NAND2x2_ASAP7_6t_L",
    "NAND3X1": "NAND3x1_ASAP7_6t_L",
    "NAND3X2": "NAND3x2_ASAP7_6t_L",
    "NOR2X1": "NOR2x1_ASAP7_6t_L",
    "NOR2X2": "NOR2x2_ASAP7_6t_L",
    "NOR3X1": "NOR3x1_ASAP7_6t_L",
    "NOR3X2": "NOR3x2_ASAP7_6t_L",
    "OR2X2": "OR2x2_ASAP7_6t_L",
    "OR2X4": "OR2x4_ASAP7_6t_L",
    "OR3X1": "OR3x1_ASAP7_6t_L",
    "OR3X2": "OR3x2_ASAP7_6t_L",
    "OR3X4": "OR3x4_ASAP7_6t_L",
    "OR4X1": "OR4x1_ASAP7_6t_L",
    "OR4X2": "OR2x2_ASAP7_6t_L",
    "XOR2X2": "XOR2x2_ASAP7_6t_L",
    "XNOR2X2": "XNOR2x2_ASAP7_6t_L",
}

ZERO_SPI_FEATS = {
    "wp_sum": 0.0,
    "wn_sum": 0.0,
    "wp_over_wn": 0.0,
}


# ======================================================
# ¹¤¾ßº¯Êý£ºSPICE ÌØÕ÷
# ======================================================

def parse_spi_features_from_text(text: str) -> Dict[str, float]:
    devs = parse_transistors_spice(text)
    feats = extract_wl_features(devs)

    # »ñÈ¡Ô­Ê¼µ¥Î» (Ã×)
    raw_wp_sum = float(feats.get("wp_sum", 0.0))
    raw_wn_sum = float(feats.get("wn_sum", 0.0))

    # ¼ÆËã±ÈÂÊ
    if raw_wn_sum > 1e-15:
        wp_over_wn = raw_wp_sum / raw_wn_sum
    else:
        wp_over_wn = 0.0

    # ×ª»»ÎªÎ¢Ã×
    wp_sum_um = raw_wp_sum * 1e6
    wn_sum_um = raw_wn_sum * 1e6

    # È¡¶ÔÊý
    wp_sum_log = wp_sum_um
    wn_sum_log = wn_sum_um

    return {
        "wp_sum": wp_sum_log,
        "wn_sum": wn_sum_log,
        "wp_over_wn": wp_over_wn,
    }


def parse_spi_features(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return dict(ZERO_SPI_FEATS)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_spi_features_from_text(text)


# ======================================================
# É¨Ãè lib
# ======================================================

def collect_libs(root: str):
    root_path = Path(root)
    if not root_path.exists():
        return []
    found = []
    for x in root_path.rglob("*.lib"):
        if x.is_file():
            found.append(str(x))
    return sorted(set(found))


# ======================================================
# ASAP7 SP ´¦Àí
# ======================================================

def choose_asap7_sp(root_or_file: str) -> str:
    p = Path(root_or_file)
    if p.is_file():
        return str(p)

    root_path = p
    if not root_path.exists():
        return ""

    for name in ["asap7sc6t_26_L_211010.sp", "asap7sc6t_26_L.sp"]:
        cand = list(root_path.rglob(name))
        if cand:
            return str(sorted(cand)[0])

    cand = list(root_path.rglob("*.sp"))
    if cand:
        return str(sorted(cand)[0])

    cand = list(root_path.rglob("*.spi"))
    if cand:
        return str(sorted(cand)[0])

    return ""


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


# ======================================================
# ¹¹½¨ÌØÕ÷ÓëÐÐ
# ======================================================

def build_src_spi_feats(src_spi_root: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    root_path = Path(src_spi_root)
    if not root_path.exists():
        return {}, {}

    feats_map: Dict[str, Dict[str, float]] = {}
    mapping: Dict[str, str] = {}

    for cell_type in TARGET_CELL_TYPES:
        rel_name = SRC_CELL_SPI_FILES.get(cell_type)
        if rel_name is None:
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        cands = list(root_path.rglob(rel_name))
        if not cands:
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        path = str(sorted(cands)[0])
        mapping[cell_type] = path
        feats_map[cell_type] = parse_spi_features(path)

    return feats_map, mapping


def build_tgt_spi_feats_from_big_sp(tgt_sp_root_or_file: str) -> Tuple[
    Dict[str, Dict[str, float]], Dict[str, str], str]:
    sp_file = choose_asap7_sp(tgt_sp_root_or_file)
    if not sp_file or not os.path.exists(sp_file):
        print(f"[warn] TGT: asap7 SP file not found under {tgt_sp_root_or_file}")
        return {}, {}, ""

    with open(sp_file, "r", encoding="utf-8", errors="ignore") as f:
        sp_text = f.read()

    feats_map: Dict[str, Dict[str, float]] = {}
    subckt_map: Dict[str, str] = {}

    for cell_type in TARGET_CELL_TYPES:
        subckt = ASAP7_CELL_SUBCKT.get(cell_type)
        if subckt is None:
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        sub_text = extract_subckt_text(sp_text, subckt)
        if not sub_text.strip():
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue
        feats_map[cell_type] = parse_spi_features_from_text(sub_text)
        subckt_map[cell_type] = subckt

    return feats_map, subckt_map, sp_file


def _build_enhanced_row(
        tech: str,
        cell_type: str,
        cell_name: str,
        from_pin: str,
        to_pin: str,
        pol: str,
        slew: float,
        cap: float,
        voltage: float,
        temp: float,
        delay: float,
        spi_feats: Dict[str, float],
) -> Dict[str, float]:
    wp_sum_um = float(spi_feats.get("wp_sum", 0.0))
    wn_sum_um = float(spi_feats.get("wn_sum", 0.0))
    wp_over_wn = float(spi_feats.get("wp_over_wn", 0.0))

    slew_val = float(slew)
    cap_val = float(cap)

    eps = 1e-15
    req_p_linear = 1.0 / max(wp_sum_um, eps) if wp_sum_um > 0 else 0.0
    req_n_linear = 1.0 / max(wn_sum_um, eps) if wn_sum_um > 0 else 0.0

    rc_p_linear = req_p_linear * cap_val
    rc_n_linear = req_n_linear * cap_val

    if pol == "rise":
        rc_eff_linear = rc_p_linear
        req_eff_linear = req_p_linear
    else:
        rc_eff_linear = rc_n_linear
        req_eff_linear = req_n_linear

    if (wp_sum_um + wn_sum_um) > 0:
        pn_balance = (wp_sum_um - wn_sum_um) / (wp_sum_um + wn_sum_um)
    else:
        pn_balance = 0.0

    inv_v = 1.0 / max(voltage, 1e-12) if voltage > 0 else 0.0
    inv_temp = 1.0 / max(temp, 1e-12) if temp != 0 else 0.0

    log_eps = 1e-9

    row = {
        "tech": tech,
        "cell_type": cell_type,
        "cell_name": cell_name,
        "from_pin": from_pin,
        "to_pin": to_pin,

        "pol": pol,
        "voltage": float(voltage),
        "temp": float(temp),
        "delay": float(delay),

        "wp_sum": float(np.log10(wp_sum_um + log_eps)),
        "wn_sum": float(np.log10(wn_sum_um + log_eps)),

        "slew": float(np.log10(slew_val + log_eps)),
        "cap": float(np.log10(cap_val + log_eps)),

        "log_slew": float(np.log10(slew_val + log_eps)),
        "log_cap": float(np.log10(cap_val + log_eps)),

        "req_p": float(np.log10(req_p_linear + log_eps)),
        "req_n": float(np.log10(req_n_linear + log_eps)),
        "rc_p": float(np.log10(rc_p_linear + log_eps)),
        "rc_n": float(np.log10(rc_n_linear + log_eps)),
        "req_eff": float(np.log10(req_eff_linear + log_eps)),
        "rc_eff": float(np.log10(rc_eff_linear + log_eps)),

        "wp_over_wn": wp_over_wn,
        "pn_balance": pn_balance,
        "is_inv": 1 if "INV" in cell_type else 0,
        "stack_pd": 1,
        "inv_v": inv_v,
        "inv_temp": inv_temp,
    }
    return row


def to_rows(tech: str, arc_dict: dict, spi_feats: Dict[str, float]):
    rows = []
    v = float(arc_dict["nom_voltage"])
    t = float(arc_dict["nom_temperature"])
    grid_slew = arc_dict["slew"]
    grid_cap = arc_dict["cap"]

    cell_type = arc_dict["cell_type"]
    cell_name = arc_dict["cell_name"]
    from_pin = arc_dict["from_pin"]
    to_pin = arc_dict["to_pin"]

    for pol, M in [("rise", arc_dict["cell_rise"]), ("fall", arc_dict["cell_fall"])]:
        for i, s in enumerate(grid_slew):
            for j, c in enumerate(grid_cap):
                delay = float(M[i, j])
                if not np.isfinite(delay):
                    continue
                if delay < -1e-6:
                    continue

                row = _build_enhanced_row(
                    tech=tech,
                    cell_type=cell_type,
                    cell_name=cell_name,
                    from_pin=from_pin,
                    to_pin=to_pin,
                    pol=pol,
                    slew=float(s),
                    cap=float(c),
                    voltage=v,
                    temp=t,
                    delay=delay,
                    spi_feats=spi_feats,
                )
                rows.append(row)
    return rows


# ======================================================
# ÇÐ·ÖÂß¼­
# ======================================================

def split_by_cell_type(df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42) -> Tuple[List[str], List[str], List[str]]:
    cell_types = df["cell_type"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(cell_types)

    n = len(cell_types)
    if n < 3:
        return cell_types.tolist(), [], []

    n_train = int(np.floor(ratios[0] * n))
    n_val = int(np.floor(ratios[1] * n))
    n_test = n - n_train - n_val

    if n_test < 1 and n > 2:
        n_test = 1
        n_train = n - n_val - n_test

    c_train = cell_types[:n_train]
    c_val = cell_types[n_train:n_train + n_val]
    c_test = cell_types[n_train + n_val:]

    return c_train.tolist(), c_val.tolist(), c_test.tolist()


# ======================================================
# ¡¾ºËÐÄÐÂÔö¡¿¼ÆËã Scalers
# ======================================================

def compute_and_save_scalers(df_train_pool: pd.DataFrame, out_dir: str):
    """
    »ùÓÚ Target Training Pool ¼ÆËãÌØÕ÷ºÍ±êÇ©µÄ¾ùÖµ/·½²î²¢±£´æ¡£
    ×¢Òâ£ºÐèÒª´¦Àí pol_bit¡£
    """
    print("[info] Computing scalers from Target Training Pool...")

    # 1. ÁÙÊ±´´½¨ pol_bit ÓÃÓÚÍ³¼Æ£¬²»ÐÞ¸ÄÔ­ df
    df_temp = df_train_pool.copy()
    if "pol_bit" not in df_temp.columns:
        if "pol" in df_temp.columns:
            df_temp["pol_bit"] = (df_temp["pol"] == "rise").astype(float)
        else:
            df_temp["pol_bit"] = 0.0

    # 2. ¼ÆËãÌØÕ÷Í³¼ÆÁ¿
    stats = {"mean": {}, "std": {}}
    for col in NUMERIC_COLS:
        if col in df_temp.columns:
            # ×ª»»Îª float ÒÔÈ·±£ JSON ¿ÉÐòÁÐ»¯
            stats["mean"][col] = float(df_temp[col].mean())
            stats["std"][col] = float(df_temp[col].std())
        else:
            print(f"[warn] Feature {col} not found in dataframe, using default 0/1.")
            stats["mean"][col] = 0.0
            stats["std"][col] = 1.0

    # 3. ¼ÆËã±êÇ©Í³¼ÆÁ¿ (Delay)
    if "delay" in df_temp.columns:
        y_mean = float(df_temp["delay"].mean())
        y_std = float(df_temp["delay"].std())
    else:
        y_mean = 0.0
        y_std = 1.0

    y_info = {"mean": y_mean, "std": y_std}

    # 4. ±£´æ
    with open(os.path.join(out_dir, "scaler_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(out_dir, "y_scaler.json"), "w") as f:
        json.dump(y_info, f, indent=2)

    print(f"[info] Scalers saved to {out_dir}")
    print(f"       Y Stats: mean={y_mean:.4f}, std={y_std:.4f}")


# ======================================================
# Ö÷Á÷³Ì
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lib", required=True, help="Nangate45 lib ¸ùÄ¿Â¼")
    parser.add_argument("--tgt_lib", required=True, help="ASAP7 lib ¸ùÄ¿Â¼")
    parser.add_argument("--src_spi", required=True, help="Nangate45 SPI ¸ùÄ¿Â¼")
    parser.add_argument("--tgt_sp", required=True, help="ASAP7 SP ¸ùÄ¿Â¼»òÎÄ¼þ")
    parser.add_argument("--out_dir", required=True, help="Êä³öÄ¿Â¼")
    parser.add_argument("--target_label_ratio", type=float, default=0.9, help="Train Set ÖÐ±£Áô¶àÉÙ±ÈÀýµÄÓÐ±êÇ©Êý¾Ý")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) ÕÒµ½ËùÓÐ lib ----------
    src_libs = collect_libs(args.src_lib)
    tgt_libs = collect_libs(args.tgt_lib)

    print(f"[info] Nangate45 libs found: {len(src_libs)}")
    print(f"[info] ASAP7 libs found    : {len(tgt_libs)}")

    if len(tgt_libs) == 0:
        raise SystemExit("[error] ASAP7 ÖÐÃ»ÓÐÕÒµ½ÈÎºÎ .lib£¬ÇëÈ·ÈÏÄ¿Â¼¡£")

    # ---------- 2) SPI ÌØÕ÷ ----------
    src_spi_feats_map, src_spi_map = build_src_spi_feats(args.src_spi)
    tgt_spi_feats_map, tgt_subckt_map, tgt_sp_file = build_tgt_spi_feats_from_big_sp(args.tgt_sp)

    # ---------- 3) ±éÀú lib ½âÎöÊý¾Ý ----------
    all_src_rows = []
    all_tgt_rows = []

    # Nangate45
    for path in src_libs:
        print(f"[info] parse SRC lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = src_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_src_rows += to_rows("Nangate45", arc, spi_feats)

    # ASAP7
    for path in tgt_libs:
        print(f"[info] parse TGT lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = tgt_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_tgt_rows += to_rows("ASAP7", arc, spi_feats)

    if len(all_tgt_rows) == 0:
        raise SystemExit("[error] ¹¹½¨Ê§°Ü£ºASAP7 Ä¿±êÓòÃ»ÓÐÈÎºÎÓÐÐ§Ñù±¾¡£")

    df_src = pd.DataFrame(all_src_rows) if len(all_src_rows) > 0 else pd.DataFrame()
    df_tgt = pd.DataFrame(all_tgt_rows)

    # È«¾Ö´òÂÒ
    df_tgt = df_tgt.sample(frac=1, random_state=42).reset_index(drop=True)
    if not df_src.empty:
        df_src = df_src.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------- 4) Êý¾Ý¼¯»®·Ö ----------
    train_cells, val_cells, test_cells = split_by_cell_type(df_tgt, ratios=(0.14, 0.14, 0.72), seed=42)

    print("\n" + "=" * 50)
    print("¡¾Êý¾Ý¼¯ÇÐ·ÖÏêÇé (By Cell Type)¡¿")
    print(f"  Train Cells ({len(train_cells)}): {train_cells}")
    print(f"  Val   Cells ({len(val_cells)}): {val_cells}")
    print(f"  Test  Cells ({len(test_cells)}): {test_cells}")
    print("=" * 50 + "\n")

    df_tgt_train_pool = df_tgt[df_tgt["cell_type"].isin(train_cells)].copy()
    df_tgt_val = df_tgt[df_tgt["cell_type"].isin(val_cells)].copy()
    df_tgt_test = df_tgt[df_tgt["cell_type"].isin(test_cells)].copy()

    # =========================================================
    # ¡¾ÐÂÔö²½Öè¡¿¼ÆËã²¢±£´æ Scalers (»ùÓÚ Target Train Pool)
    # =========================================================
    compute_and_save_scalers(df_tgt_train_pool, args.out_dir)

    # ¼ÌÐø´¦Àí Labeled / Unlabeled
    df_tgt_train_pool = df_tgt_train_pool.sample(frac=1, random_state=123).reset_index(drop=True)
    n_train_total = len(df_tgt_train_pool)
    n_train_labeled = int(n_train_total * args.target_label_ratio)

    df_tgt_train = df_tgt_train_pool.iloc[:n_train_labeled].copy()
    df_tgt_unlabeled = df_tgt_train_pool.iloc[n_train_labeled:].copy()

    df_tgt_train["is_labeled"] = 1
    df_tgt_unlabeled["is_labeled"] = 0
    df_tgt_val["is_labeled"] = 1
    df_tgt_test["is_labeled"] = 1

    # ===== Êä³ö CSV =====
    if not df_src.empty:
        df_src.to_csv(os.path.join(args.out_dir, "src_delay.csv"), index=False)

    df_tgt_train.to_csv(os.path.join(args.out_dir, "tgt_train.csv"), index=False)
    df_tgt_val.to_csv(os.path.join(args.out_dir, "tgt_val.csv"), index=False)
    df_tgt_test.to_csv(os.path.join(args.out_dir, "tgt_test.csv"), index=False)

    df_tgt_u_safe = df_tgt_unlabeled.drop(columns=["delay"], errors="ignore")
    df_tgt_u_safe.to_csv(os.path.join(args.out_dir, "tgt_unlabeled.csv"), index=False)
    df_tgt_unlabeled.to_csv(os.path.join(args.out_dir, "tgt_unlabeled_debug.csv"), index=False)

    df_tgt.to_csv(os.path.join(args.out_dir, "tgt_delay_full.csv"), index=False)

    # ÔªÊý¾Ý
    feature_cols = [
        c for c in df_tgt_train.columns
        if c not in ["delay", "tech", "is_labeled",
                     "cell_name", "from_pin", "to_pin", "group_id"]
    ]

    meta = {
        "src_spi_by_cell": src_spi_map,
        "tgt_sp_file": tgt_sp_file,
        "tgt_subckt_by_cell": tgt_subckt_map,
        "split_info": {
            "train_cells": train_cells,
            "val_cells": val_cells,
            "test_cells": test_cells,
        },
        "stats": {
            "num_src": len(df_src),
            "num_tgt_train_labeled": len(df_tgt_train),
            "num_tgt_train_unlabeled": len(df_tgt_unlabeled),
            "num_tgt_val": len(df_tgt_val),
            "num_tgt_test": len(df_tgt_test),
        },
        "feature_cols": feature_cols,
        "cell_types": TARGET_CELL_TYPES,
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[info] DONE ¡ª Êý¾Ý¼¯¹¹½¨³É¹¦£¨Strict Cell-Based Split + Scalers£©£¡")
    print(f"[info] Check output in: {args.out_dir}")


if __name__ == "__main__":
    main()