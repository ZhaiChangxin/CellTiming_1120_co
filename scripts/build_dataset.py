import argparse
import os
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from parse_lib import parse_cell_arcs
from spi2graph import parse_transistors_spice, extract_wl_features


# ======================================================
# 配置：要保留的 cell 类型
# ======================================================

TARGET_CELL_TYPES = ["INVX1", "INVX2", "NANDX1", "NORX1", "XORX1"]

# -------- 源域（Nangate）每个 cell 对应一个 SPI 文件 --------
SRC_CELL_SPI_FILES = {
    "INVX1": "INV_X1_lpe.spi",
    "INVX2": "INV_X2_lpe.spi",
    "NANDX1": "NAND2_X1_lpe.spi",
    "NORX1": "NOR2_X1_lpe.spi",
    "XORX1": "XOR2_X1_lpe.spi",
}

# -------- 目标域（ASAP7）大 SP 文件里的 subckt 名 --------
# 这里是 “规范化 cell_type” -> “ASAP7 SP 里的 cell 子电路名”
ASAP7_CELL_SUBCKT = {
    "INVX1": "INVx1_ASAP7_6t_L",
    "INVX2": "INVx2_ASAP7_6t_L",
    "NANDX1": "NAND2x1_ASAP7_6t_L",
    "NORX1": "NOR2x1_ASAP7_6t_L",
    # XORX1 在 SIMPLE lib 中对应 XOR2xp5_ASAP7_6t_L，
    # 如果 lib 里没有 XORX1 的 arc，这个映射不会被实际用到。
    "XORX1": "XOR2xp5_ASAP7_6t_L",
}

ZERO_SPI_FEATS = {
    "wp_sum": 0.0,
    "wn_sum": 0.0,
    "wp_over_wn": 0.0,
}


# ======================================================
# 工具函数：SPICE 特征
# ======================================================

def parse_spi_features_from_text(text: str) -> Dict[str, float]:
    """
    从一段 SPICE / SP 文本里抽取器件物理特征。
    """
    devs = parse_transistors_spice(text)
    feats = extract_wl_features(devs)

    wp_sum = float(feats.get("wp_sum", 0.0))
    wn_sum = float(feats.get("wn_sum", 0.0))
    wp_over_wn = float(feats.get("wp_over_wn", 0.0) if wn_sum != 0 else 0.0)

    return {
        "wp_sum": wp_sum,
        "wn_sum": wn_sum,
        "wp_over_wn": wp_over_wn,
    }


def parse_spi_features(path: str) -> Dict[str, float]:
    """
    从文件路径读取后调用 parse_spi_features_from_text，
    主要用于 Nangate45 的每个 cell 独立 .spi。
    """
    if not os.path.exists(path):
        return dict(ZERO_SPI_FEATS)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_spi_features_from_text(text)


# ======================================================
# 扫描 lib
# ======================================================

def collect_libs(root: str):
    """
    递归收集 root 下所有 .lib 文件
    返回绝对路径列表，排序去重。
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
# ASAP7 的「全库」SP（兜底用）
# ======================================================

def choose_asap7_sp(root_or_file: str) -> str:
    """
    兼容两种情况：
    1) 传进来的是目录：在目录下自动找 asap7sc6t_26_L_*.sp；
    2) 传进来的是文件：直接返回这个文件。
    """
    p = Path(root_or_file)
    if p.is_file():
        return str(p)

    root_path = p
    if not root_path.exists():
        return ""

    # 优先常见命名
    for name in ["asap7sc6t_26_L_211010.sp", "asap7sc6t_26_L.sp"]:
        cand = list(root_path.rglob(name))
        if cand:
            return str(sorted(cand)[0])

    # 否则任意 .sp
    cand = list(root_path.rglob("*.sp"))
    if cand:
        return str(sorted(cand)[0])

    # 再不行就试试 .spi
    cand = list(root_path.rglob("*.spi"))
    if cand:
        return str(sorted(cand)[0])

    return ""


# ======================================================
# 从大 SP 文件中按 subckt 名提取 netlist 文本
# ======================================================

import re

def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    """
    在一个大 SP 文件文本 sp_text 中，找到：
        .subckt <subckt_name> ...
        ...
        .ends
    之间的所有内容并返回。
    注意：asap7sc6t_26_L_211010.sp 里的 .ends 通常不带名字，所以只匹配 '.ends'。
    """
    lines = sp_text.splitlines(keepends=True)
    collecting = False
    buf = []

    # 匹配 '.subckt <name>'
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
# 源域：每个 cell 一个 SPI（Nangate45）
# ======================================================

def build_src_spi_feats(src_spi_root: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    返回：
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
            print(f"[warn] SRC: no SPI file mapping for cell {cell_type}, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        # 在目录下递归搜索这个文件名
        cands = list(root_path.rglob(rel_name))
        if not cands:
            print(f"[warn] SRC: SPI file {rel_name} for cell {cell_type} not found, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        path = str(sorted(cands)[0])
        mapping[cell_type] = path
        feats_map[cell_type] = parse_spi_features(path)
        print(f"[info] SRC: cell {cell_type} uses SPI: {path}")

    return feats_map, mapping


# ======================================================
# 目标域：从 asap7sc6t_26_L_211010.sp 中直接按 cell 提取
# ======================================================

def build_tgt_spi_feats_from_big_sp(tgt_sp_root_or_file: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], str]:
    """
    目标域 ASAP7：
    - 只给一个大的 asap7sc6t_26_L_211010.sp；
    - 不再生成单独的 sp 文件；
    - 直接在大文件中根据 subckt 名提取每个 cell 的 netlist 段并算特征。

    返回：
      feats_map    : {cell_type: spi_feats_dict}
      subckt_map   : {cell_type: subckt_name}
      sp_file_path : 使用的 SP 文件路径
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
            print(f"[warn] TGT: no subckt mapping for cell {cell_type}, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        sub_text = extract_subckt_text(sp_text, subckt)
        if not sub_text.strip():
            print(f"[warn] TGT: subckt {subckt} for cell {cell_type} not found in {sp_file}, using ZERO features.")
            feats_map[cell_type] = dict(ZERO_SPI_FEATS)
            continue

        feats_map[cell_type] = parse_spi_features_from_text(sub_text)
        subckt_map[cell_type] = subckt
        print(f"[info] TGT: cell {cell_type} uses subckt {subckt} from {sp_file}")

    return feats_map, subckt_map, sp_file


# ======================================================
# 行构造（加特征）
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
    构造一行样本，并加上一些“物理可解释”的组合特征。
    同时把 cell_type/cell_name/from_pin/to_pin 写进 CSV，方便后续训练。
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

        "pol": pol,           # rise / fall
        "slew": float(slew),
        "cap": float(cap),
        "voltage": float(voltage),
        "temp": float(temp),
        "delay": float(delay),

        "wp_over_wn": wp_over_wn,
        "wp_sum": wp_sum,
        "wn_sum": wn_sum,
        "is_inv": 1 if "INV" in cell_type else 0,
        "stack_pu": 1,  # 可以以后改成真实的堆叠数
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
    把某一条 timing arc（一个 cell 的 from_pin→to_pin）展平成多行记录。
    arc_dict 是 parse_cell_arcs 返回的一个元素。
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
# 主流程
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lib", required=True,
                        help="Nangate45 lib 根目录（会递归找 .lib）")
    parser.add_argument("--tgt_lib", required=True,
                        help="ASAP7 lib 根目录（会递归找 .lib）")
    parser.add_argument("--src_spi", required=True,
                        help="Nangate45 SPI 根目录（每个 cell 一个网表）")
    parser.add_argument("--tgt_sp", required=True,
                        help="ASAP7 SP 根目录或文件（包含 asap7sc6t_26_L_211010.sp）")
    parser.add_argument("--out_dir", required=True,
                        help="输出目录")
    parser.add_argument("--target_label_ratio", type=float, default=0.1,
                        help="ASAP7 目标域中用于有标签监督的比例")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) 找到所有 lib ----------
    src_libs = collect_libs(args.src_lib)
    tgt_libs = collect_libs(args.tgt_lib)

    print(f"[info] Nangate45 libs found: {len(src_libs)}")
    print(f"[info] ASAP7 libs found    : {len(tgt_libs)}")

    if len(tgt_libs) == 0:
        raise SystemExit("[error] ASAP7 中没有找到任何 .lib，请确认目录。")

    # ---------- 2) 源域：每个 cell 一个 SPI；目标域：从大 SP 按 cell 提取 ----------
    src_spi_feats_map, src_spi_map = build_src_spi_feats(args.src_spi)
    tgt_spi_feats_map, tgt_subckt_map, tgt_sp_file = build_tgt_spi_feats_from_big_sp(args.tgt_sp)

    # ---------- 3) 遍历所有 lib，解析 5 类 cell ----------
    all_src_rows = []
    all_tgt_rows = []

    # 源域（Nangate）
    for path in src_libs:
        print(f"[info] parse SRC lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        print(f"   [info] arcs found: {len(arcs)}")
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = src_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_src_rows += to_rows("Nangate45", arc, spi_feats)

    # 目标域（ASAP7）
    for path in tgt_libs:
        print(f"[info] parse TGT lib: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        arcs = parse_cell_arcs(text, target_cell_types=TARGET_CELL_TYPES)
        print(f"   [info] arcs found: {len(arcs)}")
        for arc in arcs:
            cell_type = arc["cell_type"]
            spi_feats = tgt_spi_feats_map.get(cell_type, ZERO_SPI_FEATS)
            all_tgt_rows += to_rows("ASAP7", arc, spi_feats)

    # ---------- 4) 保存 CSV ----------
    if len(all_tgt_rows) == 0:
        raise SystemExit("[error] 构建失败：ASAP7 目标域没有任何有效的样本。")

    if len(all_src_rows) == 0:
        print("[warn] 源域 Nangate45 没有成功解析到样本，不过目标域数据已构建，将继续保存。")

    df_src = pd.DataFrame(all_src_rows) if len(all_src_rows) > 0 else pd.DataFrame()
    df_tgt = pd.DataFrame(all_tgt_rows)

    # 打乱
    if not df_src.empty:
        df_src = df_src.sample(frac=1, random_state=42).reset_index(drop=True)
    df_tgt = df_tgt.sample(frac=1, random_state=42).reset_index(drop=True)

    # 目标域有标签 / 无标签划分
    n_lab = max(1, int(len(df_tgt) * args.target_label_ratio))
    df_tgt_l = df_tgt.iloc[:n_lab].copy()
    df_tgt_u = df_tgt.iloc[n_lab:].copy()

    df_tgt_l["is_labeled"] = 1
    df_tgt_u["is_labeled"] = 0
    if not df_src.empty:
        df_src["is_labeled"] = 1

    # 输出
    if not df_src.empty:
        df_src.to_csv(os.path.join(args.out_dir, "src_delay.csv"), index=False)
    df_tgt_l.to_csv(os.path.join(args.out_dir, "tgt_delay_labeled.csv"), index=False)
    df_tgt_u.to_csv(os.path.join(args.out_dir, "tgt_delay_unlabeled.csv"), index=False)
    df_tgt.to_csv(os.path.join(args.out_dir, "tgt_delay.csv"), index=False)

    # 特征列：去掉 label / 域标记 / 一些纯 ID 字段
    feature_cols = [
        c for c in df_tgt.columns
        if c not in ["delay", "tech", "is_labeled",
                     "cell_name", "from_pin", "to_pin"]
    ]

    meta = {
        # 源域：每个 cell 对应的独立 SPI 文件
        "src_spi_by_cell": src_spi_map,          # {cell_type: spi_path}

        # 目标域：大 SP 文件 + 每个 cell 对应的 subckt 名称
        "tgt_sp_file": tgt_sp_file,              # asap7sc6t_26_L_211010.sp 路径
        "tgt_subckt_by_cell": tgt_subckt_map,    # {cell_type: subckt_name}

        "num_src": int(len(df_src)) if not df_src.empty else 0,
        "num_tgt_l": int(len(df_tgt_l)),
        "num_tgt_u": int(len(df_tgt_u)),
        "feature_cols": feature_cols,
        "cell_types": TARGET_CELL_TYPES,
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[info] DONE — 数据集构建成功！")
    print(f"[info] feature_dim = {len(feature_cols)}")
    print(meta)


if __name__ == "__main__":
    main()
