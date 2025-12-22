# === Pythonä»£ç æä»¶: build_dataset.py (ä¸¥æ ¼ Cell-Based ååç) ===

import argparse
import os
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# åè®¾è¿äºåºæä»¶åä½ æ¬å°ç¯å¢ä¸è´
from parse_lib import parse_cell_arcs
from spi2graph import parse_transistors_spice, extract_wl_features

# ======================================================
# éç½®ï¼è¦ä¿çç cell ç±»å (å·²æ´æ°ä¸ºä½ æä¾çæ©ååè¡¨)
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

# -------- æºåï¼Nangateï¼æ¯ä¸ª cell å¯¹åºä¸ä¸ª SPI æä»¶ --------
SRC_CELL_SPI_FILES = {
    "AND2X2": "AND2_X2_lpe.spi",
    "AND2X4": "AND2_X4_lpe.spi",
    "AND3X1": "AND3_X1_lpe.spi",
    "AND3X2": "AND3_X2_lpe.spi",
    "AND3X4": "AND2_X4_lpe.spi",
    # æ³¨æï¼åæ å°ä¼¼ä¹ç¨äº AND2_X4ï¼å¦ææ¯ç¬è¯¯è¯·èªè¡ä¿®æ­£ï¼è¿éä¿çåæ ·
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

# -------- ç®æ åï¼ASAP7ï¼å¤§ SP æä»¶éç subckt å --------
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
    "OR4X2": "OR2x2_ASAP7_6t_L",  # æ³¨æï¼åæ å°è¿éç¨äº OR2x2ï¼è¯·ç¡®è®¤æ¯å¦ä¸ºç¬è¯¯ï¼è¿éä¿çåæ ·
    "XOR2X2": "XOR2x2_ASAP7_6t_L",
    "XNOR2X2": "XNOR2x2_ASAP7_6t_L",
}

ZERO_SPI_FEATS = {
    "wp_sum": 0.0,
    "wn_sum": 0.0,
    "wp_over_wn": 0.0,
}


# ======================================================
# å·¥å·å½æ°ï¼SPICE ç¹å¾
# ======================================================

def parse_spi_features_from_text(text: str) -> Dict[str, float]:
    """
    ä»ä¸æ®µ SPICE / SP ææ¬éæ½åå¨ä»¶ç©çç¹å¾ã
    """
    devs = parse_transistors_spice(text)
    feats = extract_wl_features(devs)

    # å° W/L ä»ç±³(m)è½¬æ¢ä¸ºå¾®ç±³(um)ï¼ä¸ hgat.py ä¸­çç¹å¾å¤çä¿æä¸è´
    wp_sum = float(feats.get("wp_sum", 0.0)) * 1e6
    wn_sum = float(feats.get("wn_sum", 0.0)) * 1e6
    wp_over_wn = float(feats.get("wp_over_wn", 0.0) if wn_sum != 0 else 0.0)

    return {
        "wp_sum": wp_sum,
        "wn_sum": wn_sum,
        "wp_over_wn": wp_over_wn,
    }


def parse_spi_features(path: str) -> Dict[str, float]:
    """
    ä»æä»¶è·¯å¾è¯»ååè°ç¨ parse_spi_features_from_textï¼
    ä¸»è¦ç¨äº Nangate45 çæ¯ä¸ª cell ç¬ç« .spiã
    """
    if not os.path.exists(path):
        return dict(ZERO_SPI_FEATS)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_spi_features_from_text(text)


# ======================================================
# æ«æ lib
# ======================================================

def collect_libs(root: str):
    """
    éå½æ¶é root ä¸ææ .lib æä»¶
    è¿åç»å¯¹è·¯å¾åè¡¨ï¼æåºå»éã
    """
    root_path = Path(root)
    if not root_path.exists():
        return []

    found = []
    for x in root_path.rglob("*.lib"):
        if x.is_file():
            found.append(str(x))

    return sorted(set(found))


# ======================================================
# ASAP7 çãå¨åºãSPï¼ååºç¨ï¼
# ======================================================

def choose_asap7_sp(root_or_file: str) -> str:
    """
    å¼å®¹ä¸¤ç§æåµï¼
    1) ä¼ è¿æ¥çæ¯ç®å½ï¼å¨ç®å½ä¸èªå¨æ¾ asap7sc6t_26_L_*.spï¼
    2) ä¼ è¿æ¥çæ¯æä»¶ï¼ç´æ¥è¿åè¿ä¸ªæä»¶ã
    """
    p = Path(root_or_file)
    if p.is_file():
        return str(p)

    root_path = p
    if not root_path.exists():
        return ""

    # ä¼åå¸¸è§å½å
    for name in ["asap7sc6t_26_L_211010.sp", "asap7sc6t_26_L.sp"]:
        cand = list(root_path.rglob(name))
        if cand:
            return str(sorted(cand)[0])

    # å¦åä»»æ .sp
    cand = list(root_path.rglob("*.sp"))
    if cand:
        return str(sorted(cand)[0])

    # åä¸è¡å°±è¯è¯ .spi
    cand = list(root_path.rglob("*.spi"))
    if cand:
        return str(sorted(cand)[0])

    return ""


# ======================================================
# ä»å¤§ SP æä»¶ä¸­æ subckt åæå netlist ææ¬
# ======================================================

def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    """
    å¨ä¸ä¸ªå¤§ SP æä»¶ææ¬ sp_text ä¸­ï¼æ¾å°ï¼
        .subckt <subckt_name> ...
        ...
        .ends
    ä¹é´çææåå®¹å¹¶è¿åã
    """
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
# æºåï¼æ¯ä¸ª cell ä¸ä¸ª SPIï¼Nangate45ï¼
# ======================================================

def build_src_spi_feats(src_spi_root: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    è¿åï¼
      feats_map: {cell_type: spi_feats_dict}
      mapping  : {cell_type: spi_path}
    """
    root_path = Path(src_spi_root)
    if not root_path.exists():
        print(f"[warn] src_spi root not found: {src_spi_root}")
        return {}, {}

    feats_map: Dict[str, Dict[str, float]] = {}
    mapping: Dict[str, str] = {}

    for cell_type in TARGET_CELL_TYPES:
        rel_name = SRC_CELL_SPI_FILES.get(cell_type)
        if rel_name is None:
            # print(f"[warn] SRC: no SPI file mapping for cell {cell_type}, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        cands = list(root_path.rglob(rel_name))
        if not cands:
            # print(f"[warn] SRC: SPI file {rel_name} for cell {cell_type} not found, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        path = str(sorted(cands)[0])
        mapping[cell_type] = path
        feats_map[cell_type] = parse_spi_features(path)
        # print(f"[info] SRC: cell {cell_type} uses SPI: {path}")

    return feats_map, mapping


# ======================================================
# ç®æ åï¼ä» asap7sc6t_26_L_211010.sp ä¸­ç´æ¥æ cell æå
# ======================================================

def build_tgt_spi_feats_from_big_sp(tgt_sp_root_or_file: str) -> Tuple[
    Dict[str, Dict[str, float]], Dict[str, str], str]:
    """
    ç®æ å ASAP7ï¼
    è¿åï¼
      feats_map    : {cell_type: spi_feats_dict}
      subckt_map   : {cell_type: subckt_name}
      sp_file_path : ä½¿ç¨ç SP æä»¶è·¯å¾
    """
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
            # print(f"[warn] TGT: no subckt mapping for cell {cell_type}, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        sub_text = extract_subckt_text(sp_text, subckt)
        if not sub_text.strip():
            # print(f"[warn] TGT: subckt {subckt} for cell {cell_type} not found in {sp_file}, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        feats_map[cell_type] = parse_spi_features_from_text(sub_text)
        subckt_map[cell_type] = subckt
        # print(f"[info] TGT: cell {cell_type} uses subckt {subckt} from {sp_file}")

    return feats_map, subckt_map, sp_file


# ======================================================
# è¡æé ï¼å ç¹å¾ï¼
# ======================================================

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
    """
    æé ä¸è¡æ ·æ¬ï¼å¹¶å ä¸ä¸äºâç©çå¯è§£éâçç»åç¹å¾ã
    """
    eps = 1e-12

    wp_sum = float(spi_feats.get("wp_sum", 0.0))
    wn_sum = float(spi_feats.get("wn_sum", 0.0))
    wp_over_wn = float(spi_feats.get("wp_over_wn", 0.0))

    req_p = 1.0 / max(wp_sum, eps) if wp_sum > 0 else 0.0
    req_n = 1.0 / max(wn_sum, eps) if wn_sum > 0 else 0.0

    rc_p = req_p * cap
    rc_n = req_n * cap

    if pol == "rise":
        rc_eff = rc_p
        req_eff = req_p
    else:
        rc_eff = rc_n
        req_eff = req_n

    log_slew = float(np.log1p(max(slew, 0.0)))
    log_cap = float(np.log1p(max(cap, 0.0)))

    inv_v = 1.0 / max(voltage, eps) if voltage > 0 else 0.0
    inv_temp = 1.0 / max(temp, eps) if temp != 0 else 0.0

    if (wp_sum + wn_sum) > 0:
        pn_balance = (wp_sum - wn_sum) / (wp_sum + wn_sum)
    else:
        pn_balance = 0.0

    row = {
        "tech": tech,
        "cell_type": cell_type,
        "cell_name": cell_name,
        "from_pin": from_pin,
        "to_pin": to_pin,

        "pol": pol,
        "slew": float(slew),
        "cap": float(cap),
        "voltage": float(voltage),
        "temp": float(temp),
        "delay": float(delay),

        "wp_over_wn": wp_over_wn,
        "wp_sum": wp_sum,
        "wn_sum": wn_sum,
        "is_inv": 1 if "INV" in cell_type else 0,
        # ç®åæ è¯ç¬¦ï¼ä¸ä¸å®åç¡®ï¼ä»ä½åè
        "stack_pu": 1,
        "stack_pd": 1,

        "log_slew": log_slew,
        "log_cap": log_cap,
        "req_p": req_p,
        "req_n": req_n,
        "rc_p": rc_p,
        "rc_n": rc_n,
        "rc_eff": rc_eff,
        "req_eff": req_eff,
        "inv_v": inv_v,
        "inv_temp": inv_temp,
        "pn_balance": pn_balance,
    }
    return row


def to_rows(tech: str, arc_dict: dict, spi_feats: Dict[str, float]):
    """
    ææä¸æ¡ timing arcï¼ä¸ä¸ª cell ç from_pinâto_pinï¼å±å¹³æå¤è¡è®°å½ã
    """
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
                # è¿æ»¤ææ¾ç¶éè¯¯çè´å»¶è¿ï¼æé¤ float è¯¯å·®ï¼
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
# æ°çååé»è¾ï¼Strict Cell-Based Split
# ======================================================

def split_by_cell_type(df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42) -> Tuple[List[str], List[str], List[str]]:
    """
    è¿å Train/Val/Test åå«ç cell_type åè¡¨
    """
    cell_types = df["cell_type"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(cell_types)

    n = len(cell_types)
    if n < 3:
        print(f"[WARN] åªæ {n} ç§ Cellï¼æ æ³è¿è¡ææç Train/Val/Test ååï¼")
        # ååºï¼å¨é¨æ¾å¥ Trainï¼é¿åæ¥éï¼ä½è¯ä¼°ä¼å¤±æ
        return cell_types.tolist(), [], []

    n_train = int(np.floor(ratios[0] * n))
    n_val = int(np.floor(ratios[1] * n))
    n_test = n - n_train - n_val

    # å¼ºå¶è³å°ä¿è¯ Test æ 1 ä¸ª (å¦æ Cell å¾å°)
    if n_test < 1 and n > 2:
        n_test = 1
        n_train = n - n_val - n_test

    c_train = cell_types[:n_train]
    c_val = cell_types[n_train:n_train + n_val]
    c_test = cell_types[n_train + n_val:]

    return c_train.tolist(), c_val.tolist(), c_test.tolist()


# ======================================================
# ä¸»æµç¨
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lib", required=True,
                        help="Nangate45 lib æ ¹ç®å½ï¼ä¼éå½æ¾ .libï¼")
    parser.add_argument("--tgt_lib", required=True,
                        help="ASAP7 lib æ ¹ç®å½ï¼ä¼éå½æ¾ .libï¼")
    parser.add_argument("--src_spi", required=True,
                        help="Nangate45 SPI æ ¹ç®å½ï¼æ¯ä¸ª cell ä¸ä¸ªç½è¡¨ï¼")
    parser.add_argument("--tgt_sp", required=True,
                        help="ASAP7 SP æ ¹ç®å½ææä»¶ï¼åå« asap7sc6t_26_L_211010.spï¼")
    parser.add_argument("--out_dir", required=True,
                        help="è¾åºç®å½")
    parser.add_argument("--target_label_ratio", type=float, default=0.9,
                        help="Train Set ä¸­ä¿çå¤å°æ¯ä¾çææ ç­¾æ°æ® (0.0~1.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) æ¾å°ææ lib ----------
    src_libs = collect_libs(args.src_lib)
    tgt_libs = collect_libs(args.tgt_lib)

    print(f"[info] Nangate45 libs found: {len(src_libs)}")
    print(f"[info] ASAP7 libs found    : {len(tgt_libs)}")

    if len(tgt_libs) == 0:
        raise SystemExit("[error] ASAP7 ä¸­æ²¡ææ¾å°ä»»ä½ .libï¼è¯·ç¡®è®¤ç®å½ã")

    # ---------- 2) æºå SPI ç¹å¾ & ç®æ å SP ç¹å¾ ----------
    src_spi_feats_map, src_spi_map = build_src_spi_feats(args.src_spi)
    tgt_spi_feats_map, tgt_subckt_map, tgt_sp_file = build_tgt_spi_feats_from_big_sp(args.tgt_sp)

    # ---------- 3) éåææ lib ----------
    all_src_rows = []
    all_tgt_rows = []

    # æºåï¼Nangateï¼
    for path in src_libs:
        print(f"[info] parse SRC lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        # print(f"   [info] arcs found: {len(arcs)}")
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = src_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_src_rows += to_rows("Nangate45", arc, spi_feats)

    # ç®æ åï¼ASAP7ï¼
    for path in tgt_libs:
        print(f"[info] parse TGT lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        # print(f"   [info] arcs found: {len(arcs)}")
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = tgt_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_tgt_rows += to_rows("ASAP7", arc, spi_feats)

    # ---------- 4) æ ¸å¿ï¼åºäº Cell Type çåå ----------
    if len(all_tgt_rows) == 0:
        raise SystemExit("[error] æå»ºå¤±è´¥ï¼ASAP7 ç®æ åæ²¡æä»»ä½ææçæ ·æ¬ã")

    df_src = pd.DataFrame(all_src_rows) if len(all_src_rows) > 0 else pd.DataFrame()
    df_tgt = pd.DataFrame(all_tgt_rows)

    # å¨å±æä¹±ï¼æç ´ Slew/Cap çé¡ºåº
    df_tgt = df_tgt.sample(frac=1, random_state=42).reset_index(drop=True)
    if not df_src.empty:
        df_src = df_src.sample(frac=1, random_state=42).reset_index(drop=True)

    # 1. è·å Train/Val/Test ç Cell åè¡¨
    #    è¿éæ 6:2:2 æ 7:2:1 åå Cell ç§ç±»
    train_cells, val_cells, test_cells = split_by_cell_type(df_tgt, ratios=(0.7, 0.2, 0.1), seed=42)

    print("\n" + "=" * 50)
    print("ãæ°æ®éååè¯¦æ (By Cell Type)ã")
    print(f"  Train Cells ({len(train_cells)}): {train_cells}")
    print(f"  Val   Cells ({len(val_cells)}): {val_cells}")
    print(f"  Test  Cells ({len(test_cells)}): {test_cells}")
    print("=" * 50 + "\n")

    # 2. æ ¹æ® Cell åè¡¨ç­éæ°æ®
    df_tgt_train_pool = df_tgt[df_tgt["cell_type"].isin(train_cells)].copy()
    df_tgt_val = df_tgt[df_tgt["cell_type"].isin(val_cells)].copy()
    df_tgt_test = df_tgt[df_tgt["cell_type"].isin(test_cells)].copy()

    # 3. å¤çåçç£ Labeled / Unlabeled
    #    æ³¨æï¼åªå¨ Train Set éå Maskï¼Val/Test å¿é¡»ä¿ç Label ä»¥ä¾è¯ä¼°
    df_tgt_train_pool = df_tgt_train_pool.sample(frac=1, random_state=123).reset_index(drop=True)
    n_train_total = len(df_tgt_train_pool)
    n_train_labeled = int(n_train_total * args.target_label_ratio)

    df_tgt_train = df_tgt_train_pool.iloc[:n_train_labeled].copy()  # ææ ç­¾è®­ç»é
    df_tgt_unlabeled = df_tgt_train_pool.iloc[n_train_labeled:].copy()  # æ æ ç­¾è®­ç»é

    # 4. æä¸ is_labeled æ è®°
    df_tgt_train["is_labeled"] = 1
    df_tgt_unlabeled["is_labeled"] = 0
    df_tgt_val["is_labeled"] = 1
    df_tgt_test["is_labeled"] = 1

    # ===== è¾åº =====
    if not df_src.empty:
        df_src.to_csv(os.path.join(args.out_dir, "src_delay.csv"), index=False)

    # å¯¼åºæä»¶
    df_tgt_train.to_csv(os.path.join(args.out_dir, "tgt_train.csv"), index=False)
    df_tgt_val.to_csv(os.path.join(args.out_dir, "tgt_val.csv"), index=False)
    df_tgt_test.to_csv(os.path.join(args.out_dir, "tgt_test.csv"), index=False)

    # æ æ ç­¾æ°æ®ï¼
    # 1. tgt_unlabeled.csv (ä¸å« delayï¼æ¨¡æçå®åºæ¯)
    # 2. tgt_unlabeled_debug.csv (å« delayï¼ç¨äº debug)
    df_tgt_u_safe = df_tgt_unlabeled.drop(columns=["delay"], errors="ignore")
    df_tgt_u_safe.to_csv(os.path.join(args.out_dir, "tgt_unlabeled.csv"), index=False)
    df_tgt_unlabeled.to_csv(os.path.join(args.out_dir, "tgt_unlabeled_debug.csv"), index=False)

    # å¨éæ°æ®å¤ä»½
    df_tgt.to_csv(os.path.join(args.out_dir, "tgt_delay_full.csv"), index=False)

    # ç¹å¾åè®°å½
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

    print("[info] DONE â æ°æ®éæå»ºæåï¼Strict Cell-Based Splitï¼ï¼")
    print(f"[info] Check output in: {args.out_dir}")


if __name__ == "__main__":
    main()