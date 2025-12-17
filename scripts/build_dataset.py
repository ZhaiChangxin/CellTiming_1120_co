# === Python代码文件: build_dataset.py (严格 Cell-Based 切分版) ===

import argparse
import os
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# 假设这些库文件和你本地环境一致
from parse_lib import parse_cell_arcs
from spi2graph import parse_transistors_spice, extract_wl_features

# ======================================================
# 配置：要保留的 cell 类型 (已更新为你提供的扩充列表)
# ======================================================

TARGET_CELL_TYPES = [
    "AND2X2",
    "BUFX2",
    "INVX1",
    "NAND2X1",
    "NOR2X1", "NOR2X2",
    "OR2X2", "OR2X4",
    "XNOR2X2", "XOR2X2",
]

# -------- 源域（Nangate）每个 cell 对应一个 SPI 文件 --------
SRC_CELL_SPI_FILES = {
    "AND2X2": "AND2_X2_lpe.spi",
    "BUFX2": "BUF_X2_lpe.spi",
    "INVX1": "INV_X1_lpe.spi",
    "NAND2X1": "NAND2_X1_lpe.spi",
    "NOR2X1": "NOR2_X1_lpe.spi",
    "NOR2X2": "NOR2_X2_lpe.spi",
    "OR2X2": "OR2_X2_lpe.spi",
    "OR2X4": "OR2_X4_lpe.spi",
    "XOR2X2": "XOR2_X2_lpe.spi",
    "XNOR2X2": "XNOR2_X2_lpe.spi",
}

# -------- 目标域（ASAP7）大 SP 文件里的 subckt 名 --------
ASAP7_CELL_SUBCKT = {
    "AND2X2": "AND2x2_ASAP7_6t_L",
    "BUFX2": "BUFx2_ASAP7_6t_L",
    "INVX1": "INVx1_ASAP7_6t_L",
    "NAND2X1": "NAND2x1_ASAP7_6t_L",
    "NOR2X1": "NOR2x1_ASAP7_6t_L",
    "NOR2X2": "NOR2x2_ASAP7_6t_L",
    "OR2X2": "OR2x2_ASAP7_6t_L",
    "OR2X4": "OR2x4_ASAP7_6t_L",
    "XOR2X2": "XOR2x2_ASAP7_6t_L",
    "XNOR2X2": "XNOR2x2_ASAP7_6t_L",
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

    # 将 W/L 从米(m)转换为微米(um)，与 hgat.py 中的特征处理保持一致
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

def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    """
    在一个大 SP 文件文本 sp_text 中，找到：
        .subckt <subckt_name> ...
        ...
        .ends
    之间的所有内容并返回。
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
# 目标域：从 asap7sc6t_26_L_211010.sp 中直接按 cell 提取
# ======================================================

def build_tgt_spi_feats_from_big_sp(tgt_sp_root_or_file: str) -> Tuple[
    Dict[str, Dict[str, float]], Dict[str, str], str]:
    """
    目标域 ASAP7：
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
        # 简单标识符，不一定准确，仅作参考
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
    把某一条 timing arc（一个 cell 的 from_pin→to_pin）展平成多行记录。
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
                # 过滤掉显然错误的负延迟（排除 float 误差）
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
# 新版切分逻辑：Strict Cell-Based Split
# ======================================================

def split_by_cell_type(df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed=42) -> Tuple[List[str], List[str], List[str]]:
    """
    返回 Train/Val/Test 包含的 cell_type 列表
    """
    cell_types = df["cell_type"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(cell_types)

    n = len(cell_types)
    if n < 3:
        print(f"[WARN] 只有 {n} 种 Cell，无法进行有效的 Train/Val/Test 划分！")
        # 兜底：全部放入 Train，避免报错，但评估会失效
        return cell_types.tolist(), [], []

    n_train = int(np.floor(ratios[0] * n))
    n_val = int(np.floor(ratios[1] * n))
    n_test = n - n_train - n_val

    # 强制至少保证 Test 有 1 个 (如果 Cell 很少)
    if n_test < 1 and n > 2:
        n_test = 1
        n_train = n - n_val - n_test

    c_train = cell_types[:n_train]
    c_val = cell_types[n_train:n_train + n_val]
    c_test = cell_types[n_train + n_val:]

    return c_train.tolist(), c_val.tolist(), c_test.tolist()


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
    parser.add_argument("--target_label_ratio", type=float, default=0.9,
                        help="Train Set 中保留多少比例的有标签数据 (0.0~1.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) 找到所有 lib ----------
    src_libs = collect_libs(args.src_lib)
    tgt_libs = collect_libs(args.tgt_lib)

    print(f"[info] Nangate45 libs found: {len(src_libs)}")
    print(f"[info] ASAP7 libs found    : {len(tgt_libs)}")

    if len(tgt_libs) == 0:
        raise SystemExit("[error] ASAP7 中没有找到任何 .lib，请确认目录。")

    # ---------- 2) 源域 SPI 特征 & 目标域 SP 特征 ----------
    src_spi_feats_map, src_spi_map = build_src_spi_feats(args.src_spi)
    tgt_spi_feats_map, tgt_subckt_map, tgt_sp_file = build_tgt_spi_feats_from_big_sp(args.tgt_sp)

    # ---------- 3) 遍历所有 lib ----------
    all_src_rows = []
    all_tgt_rows = []

    # 源域（Nangate）
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

    # 目标域（ASAP7）
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

    # ---------- 4) 核心：基于 Cell Type 的切分 ----------
    if len(all_tgt_rows) == 0:
        raise SystemExit("[error] 构建失败：ASAP7 目标域没有任何有效的样本。")

    df_src = pd.DataFrame(all_src_rows) if len(all_src_rows) > 0 else pd.DataFrame()
    df_tgt = pd.DataFrame(all_tgt_rows)

    # 全局打乱，打破 Slew/Cap 的顺序
    df_tgt = df_tgt.sample(frac=1, random_state=42).reset_index(drop=True)
    if not df_src.empty:
        df_src = df_src.sample(frac=1, random_state=42).reset_index(drop=True)

    # 1. 获取 Train/Val/Test 的 Cell 列表
    #    这里按 6:2:2 或 7:2:1 切分 Cell 种类
    train_cells, val_cells, test_cells = split_by_cell_type(df_tgt, ratios=(0.7, 0.2, 0.1), seed=42)

    print("\n" + "=" * 50)
    print("【数据集切分详情 (By Cell Type)】")
    print(f"  Train Cells ({len(train_cells)}): {train_cells}")
    print(f"  Val   Cells ({len(val_cells)}): {val_cells}")
    print(f"  Test  Cells ({len(test_cells)}): {test_cells}")
    print("=" * 50 + "\n")

    # 2. 根据 Cell 列表筛选数据
    df_tgt_train_pool = df_tgt[df_tgt["cell_type"].isin(train_cells)].copy()
    df_tgt_val = df_tgt[df_tgt["cell_type"].isin(val_cells)].copy()
    df_tgt_test = df_tgt[df_tgt["cell_type"].isin(test_cells)].copy()

    # 3. 处理半监督 Labeled / Unlabeled
    #    注意：只在 Train Set 里做 Mask，Val/Test 必须保留 Label 以供评估
    df_tgt_train_pool = df_tgt_train_pool.sample(frac=1, random_state=123).reset_index(drop=True)
    n_train_total = len(df_tgt_train_pool)
    n_train_labeled = int(n_train_total * args.target_label_ratio)

    df_tgt_train = df_tgt_train_pool.iloc[:n_train_labeled].copy()  # 有标签训练集
    df_tgt_unlabeled = df_tgt_train_pool.iloc[n_train_labeled:].copy()  # 无标签训练集

    # 4. 打上 is_labeled 标记
    df_tgt_train["is_labeled"] = 1
    df_tgt_unlabeled["is_labeled"] = 0
    df_tgt_val["is_labeled"] = 1
    df_tgt_test["is_labeled"] = 1

    # ===== 输出 =====
    if not df_src.empty:
        df_src.to_csv(os.path.join(args.out_dir, "src_delay.csv"), index=False)

    # 导出文件
    df_tgt_train.to_csv(os.path.join(args.out_dir, "tgt_train.csv"), index=False)
    df_tgt_val.to_csv(os.path.join(args.out_dir, "tgt_val.csv"), index=False)
    df_tgt_test.to_csv(os.path.join(args.out_dir, "tgt_test.csv"), index=False)

    # 无标签数据：
    # 1. tgt_unlabeled.csv (不含 delay，模拟真实场景)
    # 2. tgt_unlabeled_debug.csv (含 delay，用于 debug)
    df_tgt_u_safe = df_tgt_unlabeled.drop(columns=["delay"], errors="ignore")
    df_tgt_u_safe.to_csv(os.path.join(args.out_dir, "tgt_unlabeled.csv"), index=False)
    df_tgt_unlabeled.to_csv(os.path.join(args.out_dir, "tgt_unlabeled_debug.csv"), index=False)

    # 全量数据备份
    df_tgt.to_csv(os.path.join(args.out_dir, "tgt_delay_full.csv"), index=False)

    # 特征列记录
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

    print("[info] DONE — 数据集构建成功（Strict Cell-Based Split）！")
    print(f"[info] Check output in: {args.out_dir}")


if __name__ == "__main__":
    main()