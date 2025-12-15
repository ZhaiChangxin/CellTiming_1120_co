import re
import numpy as np

# ===================== 单位换算表 =====================

CAP_PREFIX_TO_PF = {
    "f": 1e-3,   # fF → 1e-3 pF
    "p": 1.0,    # pF
    "n": 1e3,    # nF
    "u": 1e6,    # uF
    "m": 1e9,    # mF
    "": 1.0,
}


# ===================== 基础工具 =====================

def _brace_block(text: str, start_idx: int):
    """找到从 text[start_idx] 开始的一整个 {...} 块并返回。"""
    assert text[start_idx] == "{"
    depth = 0
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1], i + 1
    raise ValueError("brace mismatch")


def _time_scale_to_ps(lib_text: str) -> float:
    """把 lib 里的 time_unit 换算到 ps."""
    m = re.search(r'time_unit\s*:\s*"([\d.]+)\s*([a-zA-Z]+)"', lib_text)
    if not m:
        return 1.0

    val = float(m.group(1))
    unit = m.group(2).strip().lower()

    if unit in ("s", "sec", "secs", "second", "seconds"):
        scale = 1e12
    elif unit.startswith("ms"):
        scale = 1e9
    elif unit.startswith("us"):
        scale = 1e6
    elif unit.startswith("ns"):
        scale = 1e3
    elif unit.startswith("ps"):
        scale = 1.0
    elif unit.startswith("fs"):
        scale = 1e-3
    else:
        scale = 1.0

    return val * scale


def _cap_scale_to_pF(lib_text: str) -> float:
    """把 lib 里的电容单位换算到 pF."""
    m = re.search(
        r'capacitive_load_unit\s*\(\s*([\d.]+)\s*,\s*([fpnumk]?f)\s*\)',
        lib_text, re.I
    )
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        prefix = unit[0] if len(unit) > 1 else ""
        scale = CAP_PREFIX_TO_PF.get(prefix, 1.0)
        return scale / max(val, 1e-12)

    m = re.search(
        r'capacitance_unit\s*:\s*"([\d.]+)\s*([fpnumk]?f)"',
        lib_text, re.I
    )
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        prefix = unit[0] if len(unit) > 1 else ""
        scale = CAP_PREFIX_TO_PF.get(prefix, 1.0)
        return scale / max(val, 1e-12)

    return 1.0


def _parse_nominal_conditions(lib_text: str):
    """解析 nom_voltage / nom_temperature."""
    m_v = re.search(r'nom_voltage\s*:\s*([-+]?\d*\.?\d+)', lib_text)
    if m_v:
        nom_v = float(m_v.group(1))
    else:
        m_v2 = re.search(r'voltage\s*:\s*([-+]?\d*\.?\d+)', lib_text)
        nom_v = float(m_v2.group(1)) if m_v2 else 1.0

    m_t = re.search(r'nom_temperature\s*:\s*([-+]?\d*\.?\d+)', lib_text)
    if m_t:
        nom_t = float(m_t.group(1))
    else:
        nom_t = 25.0

    return nom_v, nom_t


def _parse_index_list_from(block: str):
    """从 timing 块中解析 index_1 / index_2。"""
    m1 = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)\s*;', block)
    if not m1:
        m1 = re.search(r'index_1\s*:\s*"([^"]+)"\s*;', block)
    idx1 = None
    if m1:
        idx1 = [float(x.strip()) for x in m1.group(1).split(",") if x.strip()]

    m2 = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)\s*;', block)
    if not m2:
        m2 = re.search(r'index_2\s*:\s*"([^"]+)"\s*;', block)
    idx2 = None
    if m2:
        idx2 = [float(x.strip()) for x in m2.group(1).split(",") if x.strip()]

    if idx1 is None or idx2 is None:
        raise ValueError("Missing index_1 / index_2 in timing table.")

    return np.array(idx1, dtype=np.float64), np.array(idx2, dtype=np.float64)


def _parse_values_block(block: str) -> np.ndarray:
    """解析 values(...) 的 2D 数组。"""
    m = re.search(r'values\s*\(', block, re.I)
    sub = block[m.start():] if m else block

    rows = []
    for m2 in re.finditer(r'"([^"]+)"', sub, re.S):
        row_str = m2.group(1)
        vals = [float(x.strip()) for x in row_str.split(",") if x.strip()]
        if vals:
            rows.append(vals)

    if not rows:
        nums = [float(x) for x in re.findall(
            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', sub)]
        if nums:
            rows = [nums]

    return np.array(rows, dtype=np.float64)


# ===================== cell / pin / timing 搜索 =====================

def _find_all_cells(lib_text: str):
    cells = []
    for m in re.finditer(r'cell\s*\(\s*([^)]+)\)\s*\{', lib_text):
        name = m.group(1).strip().strip('"')
        block, _ = _brace_block(lib_text, m.end() - 1)
        cells.append((name, block))
    return cells


def _find_all_output_pins(cell_block: str):
    pins = []
    for m in re.finditer(r'pin\s*\(\s*"?(?P<name>[^")]+)"?\s*\)\s*\{', cell_block):
        name = m.group("name").strip()
        block, _ = _brace_block(cell_block, m.end() - 1)
        if re.search(r'direction\s*:\s*output', block, re.I):
            pins.append((name, block))
    return pins


def _find_all_timing_arcs(pin_block: str):
    arcs = []
    for m in re.finditer(r'timing\s*\(?\s*\)?\s*\{', pin_block):
        body, _ = _brace_block(pin_block, m.end() - 1)
        rp = re.search(r'related_pin\s*:\s*"([^"]+)"\s*;', body)
        if rp:
            arcs.append((rp.group(1).strip(), body))
    return arcs


# ===================== 关键：只保留 5 类 & 2 输入 =====================

def canonical_cell_type(cell_name: str):
    """
    归一化 cell_name → {INVX1, INVX2, NANDX1, NORX1, XORX1}

    支持：
    - Nangate45:  INV_X1 / INV_X2
    - ASAP7:      INVx1_ASAP7_... / INVx2_ASAP7_...

    规则：
    - INV：严格区分 x1/x2，不误匹配 INV_X16 / TINV_X1
    - NAND/NOR/XOR：只接受 2 输入（即 NAND2 / NOR2 / XOR2）
    - 排除 XNOR
    """

    n = cell_name.upper().replace('"', '').replace(" ", "")

    # ====================================================
    # 1) 处理 INV 系列
    # ====================================================

    # ---------- Nangate45 ----------
    # INV_X1 / INV_X2
    if re.fullmatch(r"INV_X1", n):
        return "INVX1"
    if re.fullmatch(r"INV_X2", n):
        return "INVX2"

    # ---------- ASAP7 ----------
    # INVx1_ASAP7_6t_L  → uppercase 后是 INVX1_ASAP7_6T_L
    if n.startswith("INVX1_ASAP7") or "_INVX1_" in n:
        return "INVX1"
    if n.startswith("INVX2_ASAP7") or "_INVX2_" in n:
        return "INVX2"

    # ====================================================
    # 2) NAND2 → NANDX1
    # ====================================================
    # 支持 Nangate & ASAP7，如:
    #   NAND2_X1
    #   NAND2x1_ASAP7_6t_L
    if n.startswith("NAND2") and "X1" in n:
        return "NANDX1"

    # ====================================================
    # 3) NOR2 → NORX1（排除 XNOR）
    # ====================================================
    if n.startswith("NOR2") and "X1" in n and "XNOR" not in n:
        return "NORX1"

    # ====================================================
    # 4) XOR2 → XORX1（排除 XNOR）
    # ====================================================
    if n.startswith("XOR2") and "X1" in n and "XNOR" not in n:
        return "XORX2"

    return None

# ===================== 主解析接口 =====================

def parse_cell_arcs(lib_text: str, target_cell_types=None):
    if target_cell_types is not None:
        target_cell_types = set(target_cell_types)

    nom_v, nom_t = _parse_nominal_conditions(lib_text)
    t_scale = _time_scale_to_ps(lib_text)
    c_scale = _cap_scale_to_pF(lib_text)

    results = []

    for cell_name, cell_block in _find_all_cells(lib_text):
        ctype = canonical_cell_type(cell_name)
        if ctype is None:
            continue
        if (target_cell_types is not None) and (ctype not in target_cell_types):
            continue

        out_pins = _find_all_output_pins(cell_block)
        if not out_pins:
            continue

        for to_pin, pin_block in out_pins:
            arcs = _find_all_timing_arcs(pin_block)
            for from_pin, tblock in arcs:
                try:
                    mcr = re.search(r'cell_rise\s*(\([^{}]*\))?\s*\{', tblock)
                    if not mcr:
                        continue
                    cr_block, _ = _brace_block(tblock, mcr.end() - 1)
                    idx1, idx2 = _parse_index_list_from(cr_block)
                    cr_vals = _parse_values_block(cr_block)

                    mcf = re.search(r'cell_fall\s*(\([^{}]*\))?\s*\{', tblock)
                    if not mcf:
                        continue
                    cf_block, _ = _brace_block(tblock, mcf.end() - 1)
                    cf_vals = _parse_values_block(cf_block)

                    mrt = re.search(r'rise_transition\s*(\([^{}]*\))?\s*\{', tblock)
                    if mrt:
                        rt_block, _ = _brace_block(tblock, mrt.end() - 1)
                        rt_vals = _parse_values_block(rt_block)
                    else:
                        rt_vals = np.zeros_like(cr_vals)

                    mft = re.search(r'fall_transition\s*(\([^{}]*\))?\s*\{', tblock)
                    if mft:
                        ft_block, _ = _brace_block(tblock, mft.end() - 1)
                        ft_vals = _parse_values_block(ft_block)
                    else:
                        ft_vals = np.zeros_like(cf_vals)

                    idx1_ps = idx1 * t_scale
                    idx2_pf = idx2 * c_scale

                    cell_rise = cr_vals * t_scale
                    cell_fall = cf_vals * t_scale
                    rise_tr = rt_vals * t_scale
                    fall_tr = ft_vals * t_scale

                    results.append({
                        "cell_type": ctype,
                        "cell_name": cell_name,
                        "from_pin": from_pin,
                        "to_pin": to_pin,
                        "slew": idx1_ps,
                        "cap": idx2_pf,
                        "cell_rise": cell_rise,
                        "cell_fall": cell_fall,
                        "rise_transition": rise_tr,
                        "fall_transition": fall_tr,
                        "nom_voltage": nom_v,
                        "nom_temperature": nom_t,
                        "time_unit": "ps",
                        "cap_unit": "pF",
                    })
                except Exception:
                    continue

    return results


def parse_inv_arc_A_Y_auto(lib_text: str):
    """方便取一个 INVX1 的 A→Y arc，当作单位延迟参考。"""
    arcs = parse_cell_arcs(lib_text, target_cell_types=["INVX1"])
    if not arcs:
        raise ValueError("No INVX1-like cell found in lib.")

    a0 = arcs[0]
    return {
        "slew": a0["slew"],
        "cap": a0["cap"],
        "cell_rise": a0["cell_rise"],
        "cell_fall": a0["cell_fall"],
        "rise_transition": a0["rise_transition"],
        "fall_transition": a0["fall_transition"],
        "nom_voltage": a0["nom_voltage"],
        "nom_temperature": a0["nom_temperature"],
        "time_unit": "ps",
        "cap_unit": "pF",
        "cell_type": a0["cell_type"],
        "cell_name": a0["cell_name"],
        "from_pin": a0["from_pin"],
        "to_pin": a0["to_pin"],
    }
