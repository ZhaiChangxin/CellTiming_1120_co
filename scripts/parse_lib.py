
import re
import numpy as np
# ==== BEGIN: unit helpers (time→ps, cap→pF) ====
UNIT = {
    "s": 1e12, "ms": 1e9, "us": 1e6, "ns": 1e3, "ps": 1.0, "fs": 1e-3,  # time → ps
}

# 一些库可能写成 capacitive_load_unit:(1,pf) / capacitance_unit:"1fF"
CAP_PREFIX = {"f": 1e-3, "p": 1.0, "n": 1e3, "u": 1e6, "m": 1e9, "": 1.0}  # prefix to pF

def _time_scale_to_ps(lib_text: str) -> float:
    import re
    m = re.search(r'time_unit\s*:\s*"([\d.]+)\s*([a-zA-Z]+)"', lib_text)
    if not m:
        return 1.0  # 默认为 ps
    val, unit = float(m.group(1)), m.group(2).lower()
    return val * UNIT.get(unit, 1.0)

def _cap_scale_to_pF(lib_text: str) -> float:
    import re
    # 形式1：capacitive_load_unit : (1, pf);
    m = re.search(r'capacitive_load_unit\s*:\s*\(\s*([\d.]+)\s*,\s*([a-zA-Z]+)\s*\)', lib_text)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        return val * CAP_PREFIX.get(unit[0], 1.0)

    # 形式2：capacitance_unit : "1fF"
    m = re.search(r'capacitance_unit\s*:\s*"([\d.]+)\s*([a-zA-Z]+)"', lib_text)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        # 这里 unit 一般是 "ff/pf/nf" 等，取首字母做前缀
        return val * CAP_PREFIX.get(unit[0], 1.0)

    return 1.0
# ==== END: unit helpers ====

def _brace_block(text, start_idx):
    assert text[start_idx] == '{'
    depth = 1
    i = start_idx + 1
    while i < len(text):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start_idx+1:i], i+1
        i += 1
    raise ValueError("Unbalanced braces.")

def _find_cells(text):
    return [(m.group(1), m.start(), m.end()) for m in re.finditer(r'cell\s*\(\s*([^)]+)\s*\)\s*\{', text)]

def _find_pin_block(cell_block, pin_names=("Y","ZN")):
    for pin in pin_names:
        m = re.search(r'pin\s*\(\s*'+re.escape(pin)+r'\s*\)\s*\{', cell_block)
        if m:
            body, _ = _brace_block(cell_block, m.end()-1)
            return pin, body
    return None, None

def _find_timing_related_A(pin_block):
    for m in re.finditer(r'timing\s*\(\s*\)\s*\{', pin_block):
        body, _ = _brace_block(pin_block, m.end()-1)
        rp = re.search(r'related_pin\s*:\s*"([^"]+)"\s*;', body)
        if rp and rp.group(1).strip() == "A":
            return body
    return None

def _parse_values_block(block):
    rows = []
    for m in re.finditer(r'"([^"]+)"', block, re.S):
        row_str = m.group(1)
        vals = [float(x.strip()) for x in row_str.split(",") if x.strip()]
        rows.append(vals)
    if not rows:
        cleaned = block.replace("\\\\", " ").replace("\n", " ")
        m = re.search(r'values\s*\((.*)\)', cleaned, re.I)
        if m:
            inside = m.group(1)
            nums = [float(x) for x in re.findall(r'[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', inside)]
            rows = [nums]
    return np.array(rows, dtype=np.float64)

def _parse_index_list_from(block):
    # Try formats: index_1 ("..."); or index_1 : "...";
    m = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)\s*;', block)
    if not m:
        m = re.search(r'index_1\s*:\s*"([^"]+)"\s*;', block)
    idx1 = [float(x.strip()) for x in m.group(1).split(",")] if m else None

    m = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)\s*;', block)
    if not m:
        m = re.search(r'index_2\s*:\s*"([^"]+)"\s*;', block)
    idx2 = [float(x.strip()) for x in m.group(1).split(",")] if m else None
    return idx1, idx2

def parse_inv_arc_A_Y_auto(lib_text):
    nom_voltage = None
    m = re.search(r'(?m)^\s*nom_voltage\s*:\s*([0-9.]+)\s*;', lib_text)
    if m: nom_voltage = float(m.group(1))
    nom_temp = None
    m = re.search(r'(?m)^\s*nom_temperature\s*:\s*([\-0-9.]+)\s*;', lib_text)
    if m: nom_temp = float(m.group(1))

    candidates = _find_cells(lib_text)
    chosen_timing = None
    for cname, s, e in candidates:
        body, _ = _brace_block(lib_text, lib_text.find('{', s))
        pin_name, pblk = _find_pin_block(body, ("Y","ZN"))
        if not pblk: continue
        tblock = _find_timing_related_A(pblk)
        if tblock:
            chosen_timing = tblock
            break
    if chosen_timing is None:
        raise ValueError("Cannot locate inverter timing block with pin Y/ZN and related_pin A.")

    # Find cell_rise block and parse indices from there
    mcr = re.search(r'cell_rise\s*\([^)]*\)\s*\{', chosen_timing)
    if not mcr:
        mcr = re.search(r'cell_rise\s*\{', chosen_timing)
    if not mcr:
        raise ValueError("Missing cell_rise block.")
    cr_body, _ = _brace_block(chosen_timing, mcr.end()-1)
    idx1, idx2 = _parse_index_list_from(cr_body)
    if (idx1 is None) or (idx2 is None):
        raise ValueError("Missing index_1/index_2 inside cell_rise.")
    idx1 = np.array(idx1, dtype=np.float64)
    idx2 = np.array(idx2, dtype=np.float64)

    def grab(name):
        m = re.search(r'%s\s*\([^)]*\)\s*\{' % name, chosen_timing)
        if not m:
            m = re.search(r'%s\s*\{' % name, chosen_timing)
        if not m:
            raise ValueError(f"Missing table {name}.")
        body, _ = _brace_block(chosen_timing, m.end()-1)
        return _parse_values_block(body)

    cell_rise = grab("cell_rise")
    cell_fall = grab("cell_fall")
    rise_tr = grab("rise_transition")
    fall_tr = grab("fall_transition")
    # ==== BEGIN: unify units ====
    ts = _time_scale_to_ps(lib_text)   # 时间统一为 ps
    cs = _cap_scale_to_pF(lib_text)    # 电容统一为 pF

    idx1 = idx1 * ts            # slew → ps
    idx2 = idx2 * cs            # cap  → pF

    cell_rise *= ts             # delay → ps
    cell_fall *= ts
    rise_tr   *= ts             # transition → ps
    fall_tr   *= ts
    # ==== END: unify units ====
    return {
        "slew": idx1, "cap": idx2,
        "cell_rise": cell_rise, "cell_fall": cell_fall,
        "rise_transition": rise_tr, "fall_transition": fall_tr,
        "nom_voltage": nom_voltage, "nom_temperature": nom_temp,
        "time_unit": "ps", "cap_unit": "pF"
    }
