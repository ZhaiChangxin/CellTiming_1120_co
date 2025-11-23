import re
import numpy as np

# ===================== 单位换算表 =====================
# 这里就相当于在 C 里定义两个查表用的 map（不过 Python 用 dict）

# 电容单位：统一到 pF
# key 是前缀，value 是把这个前缀对应的电容转换到 pF 的比例
# 比如 ff -> 1e-3 pF, pf -> 1 pF
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
    """
    功能：给定一个字符串 text，和一个位置 start_idx，
         已知 text[start_idx] == '{'，
         从这里往后扫描，找到和这个 '{' 匹配的 '}'，
         把中间这整块内容（包括最外面的 { }）切出来。

    类比 C 里的做法：用一个计数器 depth 遍历字符串，
    遇到 '{' depth++，遇到 '}' depth--，当 depth 从 1 变回 0 时，
    就找到了匹配的结束位置。

    返回：
      (block_str, end_pos)
      block_str: 从这个 '{' 开始到匹配的 '}' 结束这段子串
      end_pos  : 结束位置的后一个索引（方便后面继续往下扫）
    """
    assert text[start_idx] == "{"
    depth = 0
    # i 相当于 C 里 for (int i = start_idx; i < len; ++i)
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            # depth 回到 0，说明外层这一对大括号匹配完成
            if depth == 0:
                # Python 的切片 [start_idx : i+1]，不含右端点 i+1
                return text[start_idx:i + 1], i + 1
    # 如果 for 跑完还没找到 depth==0，说明花括号不匹配
    raise ValueError("brace mismatch")


def _time_scale_to_ps(lib_text: str) -> float:
    """
    功能：从 .lib 里面解析 time_unit，比如：
        time_unit : "1ns";
        time_unit : "1ps";
        time_unit : "0.001ns";
    目的是把它转换成「乘到 ns 表里的值，变成 ps」的缩放因子。

    例如：
      time_unit : "1ns";  -> 返回 1ns = 1000 ps，因此返回 1000
      time_unit : "1ps";  -> 返回 1
      time_unit : "1fs";  -> 返回 0.001
    """
    # 用正则表达式在整个 lib 文本中查找 time_unit 这一行
    # r'...': 原始字符串，避免转义
    # () 里是要捕获的分组 group(1) group(2)
    m = re.search(r'time_unit\s*:\s*"([\d.]+)\s*([a-zA-Z]+)"', lib_text)
    if not m:
        # 如果没有写 time_unit，就当作 ps，缩放因子=1
        return 1.0

    # m.group(1) 对应 ([\d.]+) 这部分，比如 "1"
    val = float(m.group(1))
    # m.group(2) 对应 ([a-zA-Z]+) 比如 "ns"
    unit = m.group(2).strip().lower()  # 比如 "ns", "ps"

    # 下面这段就是 if-else 映射，不同单位对应不同 ps 缩放
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
        # 未知单位就保守一点，当成 ps
        scale = 1.0

    # 最后返回 val * scale，也就是 "1ns" 或 "0.5ns" 这种形式的实际换算
    return val * scale


def _cap_scale_to_pF(lib_text: str) -> float:
    """
    功能：解析电容单位，把 lib 中的电容单位统一转换到 pF。
    支持两种写法：
      - capacitive_load_unit (1,pf)
      - capacitance_unit : "1fF"

    思路：
    1. 先找 capacitive_load_unit(...) 这种形式
         里面有 (数值, 单位)，例如 (1,ff) / (1,pf)
       表示 lib 里的值 * 数值 * 单位 = 实际电容
    2. 如果找不到，再找 capacitance_unit:"1fF" 这种形式
    3. 根据单位前缀（f/p/n/u/m）查 CAP_PREFIX_TO_PF 映射表
    """
    # 匹配 capacitive_load_unit(1,pf) 之类
    m = re.search(
        r'capacitive_load_unit\s*\(\s*([\d.]+)\s*,\s*([fpnumk]?f)\s*\)',
        lib_text, re.I
    )
    if m:
        val = float(m.group(1))      # 比如 1
        unit = m.group(2).lower()    # 比如 pf / ff
        prefix = unit[0] if len(unit) > 1 else ""
        # 从表里查出前缀对应的系数（相对于 pF）
        scale = CAP_PREFIX_TO_PF.get(prefix, 1.0)
        # 返回：每 1 个“lib 里的电容单位值”，对应多少 pF
        # lib 值 * val * （前缀对应的F） -> pF
        # 这里做的是：pF / (val * 单位)
        return scale / max(val, 1e-12)

    # 匹配 capacitance_unit : "1fF" 这种形式
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

    # 如果都没有写，就当作 pF，缩放因子=1
    return 1.0


def _parse_nominal_conditions(lib_text: str):
    """
    功能：从 lib 里解析 nom_voltage / nom_temperature 这两个标称条件。
    类似：
        nom_voltage : 1.1;
        nom_temperature : 25;
    如果没有 nom_voltage，就尝试找 voltage。
    """
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
    """
    功能：在一个 timing 表的 block 中，解析 index_1 / index_2，
          分别对应 slew / cap 的网格坐标。

    block 大概长这样：

        cell_rise(...) {
            index_1 ("0.001, 0.01, 0.02");
            index_2 ("0.01, 0.05, 0.1");
            values ("...", "...", ...);
        }

    返回：
      idx1: numpy array, index_1 里的浮点数列表
      idx2: numpy array, index_2 里的浮点数列表
    """
    # 先找 index_1("...") 或 index_1:"..."
    m1 = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)\s*;', block)
    if not m1:
        m1 = re.search(r'index_1\s*:\s*"([^"]+)"\s*;', block)
    idx1 = None
    if m1:
        # m1.group(1) 是引号里面的字符串，类似 "0.001, 0.01, 0.02"
        # 用逗号分割并转成 float
        idx1 = [float(x.strip()) for x in m1.group(1).split(",") if x.strip()]

    # 同样方式解析 index_2
    m2 = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)\s*;', block)
    if not m2:
        m2 = re.search(r'index_2\s*:\s*"([^"]+)"\s*;', block)
    idx2 = None
    if m2:
        idx2 = [float(x.strip()) for x in m2.group(1).split(",") if x.strip()]

    if idx1 is None or idx2 is None:
        raise ValueError("Missing index_1/index_2 in timing table.")

    # 转成 numpy 数组方便后面运算
    return np.array(idx1, dtype=np.float64), np.array(idx2, dtype=np.float64)


def _parse_values_block(block: str) -> np.ndarray:
    """
    功能：只解析 cell_rise/cell_fall 里的 values(...) 数组，
          不把 index_1 / index_2 里的数字误当成值表。

    思路：
      1. 先找到 "values(" 出现的位置，只从这里往后看
      2. 在这段子串中找所有 "..."，每一个 "..." 是一行
      3. 行里用逗号分隔，转成浮点数

    返回：
      2D numpy 数组，形状大概是 [len(index_1), len(index_2)]
    """
    # 找到 "values(" 的位置
    m = re.search(r'values\s*\(', block, re.I)
    # 如果找到了，就从 values 开始截取；否则就整个 block（兜底）
    sub = block[m.start():] if m else block

    rows = []
    # 在 sub 里找所有 "xxx" 形式的字符串
    for m2 in re.finditer(r'"([^"]+)"', sub, re.S):
        row_str = m2.group(1)  # 引号里的内容
        vals = [float(x.strip()) for x in row_str.split(",") if x.strip()]
        if vals:
            rows.append(vals)

    if not rows:
        # 如果一个 "..." 都没找到，就兜底：直接提取所有数字
        nums = [float(x) for x in re.findall(
            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', sub)]
        if nums:
            rows = [nums]

    return np.array(rows, dtype=np.float64)


# ===================== cell / pin / timing 搜索 =====================

def _find_all_cells(lib_text: str):
    """
    功能：在整个 .lib 文本中找所有 cell 定义块。

    lib 里面形如：
        cell (INV_X1) {
            ...
        }
        cell (NAND2_X1) {
            ...
        }

    我们用正则找 'cell(xxx){'，然后用 _brace_block 把每个 cell 的大括号内容截出来。

    返回列表：[(cell_name, cell_block_str), ...]
    """
    cells = []
    # re.finditer 会找到所有匹配 "cell( ... ){" 的位置
    for m in re.finditer(r'cell\s*\(\s*([^)]+)\)\s*\{', lib_text):
        # group(1) 是 cell 里括号中的名字，比如 INV_X1
        name = m.group(1).strip().strip('"')
        # m.end()-1 是 '{' 的位置，从这里开始用 _brace_block 找整块
        block, _ = _brace_block(lib_text, m.end() - 1)
        cells.append((name, block))
    return cells


def _find_all_output_pins(cell_block: str):
    """
    功能：在一个 cell 的 block 里，找到所有 direction: output 的 pin。

    一个 cell 里通常有：
        pin(A) { direction : input; ... }
        pin(Y) { direction : output; ... }

    我们需要拿到输出 pin（比如 Y）的信息，去里面找 timing{} 表。

    返回列表：[(pin_name, pin_block_str), ...]
    """
    pins = []
    # 匹配 pin(...) { ... } 结构
    for m in re.finditer(r'pin\s*\(\s*"?(?P<name>[^")]+)"?\s*\)\s*\{', cell_block):
        name = m.group("name").strip()
        block, _ = _brace_block(cell_block, m.end() - 1)
        # 只有 direction : output 的才是输出脚
        if re.search(r'direction\s*:\s*output', block, re.I):
            pins.append((name, block))
    return pins


def _find_all_timing_arcs(pin_block: str):
    """
    功能：在 pin block 里找到所有 timing{} 块，并解析 related_pin。
    即：对每一个输出 pin，找所有相应的 timing 弧（from_pin -> 这个 pin）。

    例如：
        pin(Y) {
            direction : output;
            timing() {
                related_pin : "A";
                ...
            }
            timing() {
                related_pin : "B";
                ...
            }
        }

    返回：[(from_pin, timing_block_str), ...]
    """
    arcs = []
    # 找所有 timing { ... } 的开头
    for m in re.finditer(r'timing\s*\(?\s*\)?\s*\{', pin_block):
        body, _ = _brace_block(pin_block, m.end() - 1)
        # 在 timing 块中找 related_pin : "A";
        rp = re.search(r'related_pin\s*:\s*"([^"]+)"\s*;', body)
        if rp:
            arcs.append((rp.group(1).strip(), body))
    return arcs


# ===================== 5 类 cell 类型归一 =====================

def canonical_cell_type(cell_name: str):
    """
    功能：把 lib 里的 cell_name 归一成我们想用的 5 种 cell_type：
      INVX1, INVX2, NANDX1, NORX1, XORX1

    例如：
      "INV_X1"   -> "INVX1"
      "NAND2_X1" -> "NANDX1"
      "NOR4_X1"  -> "NORX1"
      "XOR2_X1"  -> "XORX1"

    这里是非常粗暴的字符串匹配（大写后查包含关系）。
    """
    n = cell_name.upper().replace('"', '').replace(" ", "")

    # 排除 INVBUF 之类（里面有 BUF 的）
    if "INV" in n and "BUF" not in n:
        # INVX1: 兼容 INV_X1 / INV1_X1 / INVX1 等，要求有 X1 且没有 X2
        if "X1" in n and "X2" not in n:
            return "INVX1"
        # INVX2: 有 INV 且有 X2
        if "X2" in n:
            return "INVX2"

    # NANDX1: 兼容 NAND2_X1, NAND_X1, NANDX1 等
    if "NAND" in n and "X1" in n:
        return "NANDX1"

    # NORX1
    if "NOR" in n and "X1" in n:
        return "NORX1"

    # XORX1
    if "XOR" in n and "X1" in n:
        return "XORX1"

    # 不在这 5 类里的，就返回 None（表示我们不关心）
    return None


# ===================== 主接口：解析多个 cell、多条 arc =====================

def parse_cell_arcs(lib_text: str,
                    target_cell_types=None):
    """
    通用解析函数：扫描整个 lib，找出指定 cell_type 的所有 timing arc。

    每条 arc 对应一个 from_pin -> to_pin（输入 -> 输出），
    我们从中拿到：
      - slew  网格（index_1）
      - cap   网格（index_2）
      - cell_rise / cell_fall / rise_transition / fall_transition
      - nom_voltage / nom_temperature
      - cell_type / cell_name / from_pin / to_pin

    返回：
      一个列表，每个元素是 dict，结构形如：

      {
        "cell_type": "INVX1"/"INVX2"/"NANDX1"/"NORX1"/"XORX1",
        "cell_name": 原始 cell 名，比如 "INV_X1",
        "from_pin":  输入 pin 名，比如 "A",
        "to_pin":    输出 pin 名，比如 "Y",

        "slew":   1D array (ps),
        "cap":    1D array (pF),
        "cell_rise": 2D (ps),
        "cell_fall": 2D (ps),
        "rise_transition": 2D (ps),
        "fall_transition": 2D (ps),
        "nom_voltage": float,
        "nom_temperature": float,
        "time_unit": "ps",
        "cap_unit": "pF",
      }
    """
    # 如果传了 target_cell_types，就转成 set，方便判断
    if target_cell_types is not None:
        target_cell_types = set(target_cell_types)

    # 解析库的标称电压/温度
    nom_v, nom_t = _parse_nominal_conditions(lib_text)
    # 得到时间和电容单位转换到 ps / pF 的比例
    t_scale = _time_scale_to_ps(lib_text)
    c_scale = _cap_scale_to_pF(lib_text)

    results = []

    # 1. 遍历所有 cell
    for cell_name, cell_block in _find_all_cells(lib_text):
        ctype = canonical_cell_type(cell_name)
        # 不在我们关心的 5 类里就跳过
        if ctype is None:
            continue
        # 如果指定了目标 cell_type 且这个不在里面，也跳过
        if (target_cell_types is not None) and (ctype not in target_cell_types):
            continue

        # 2. 找出这个 cell 里的所有输出 pin
        out_pins = _find_all_output_pins(cell_block)
        if not out_pins:
            continue

        # 3. 对每个输出 pin，找它的所有 timing arcs
        for to_pin, pin_block in out_pins:
            arcs = _find_all_timing_arcs(pin_block)
            for from_pin, tblock in arcs:
                try:
                    # ---- 解析 cell_rise ----
                    # 先找到 "cell_rise{...}" 的起点
                    mcr = re.search(r'cell_rise\s*(\([^{}]*\))?\s*\{', tblock)
                    if not mcr:
                        # 没有 cell_rise 表，直接跳过这条 arc
                        continue
                    # 从 "cell_rise" 后面的 '{' 开始，用 _brace_block 把整个块截出来
                    cr_block, _ = _brace_block(tblock, mcr.end() - 1)
                    # 从 cell_rise 块里解析 index_1/index_2
                    idx1, idx2 = _parse_index_list_from(cr_block)
                    # 从 cell_rise 块里解析 values(...)，得到 2D 表
                    cr_vals = _parse_values_block(cr_block)

                    # ---- 解析 cell_fall ----
                    mcf = re.search(r'cell_fall\s*(\([^{}]*\))?\s*\{', tblock)
                    if not mcf:
                        continue
                    cf_block, _ = _brace_block(tblock, mcf.end() - 1)
                    cf_vals = _parse_values_block(cf_block)

                    # ---- 解析 rise_transition（上升边的输出 slew）----
                    mrt = re.search(r'rise_transition\s*(\([^{}]*\))?\s*\{', tblock)
                    if mrt:
                        rt_block, _ = _brace_block(tblock, mrt.end() - 1)
                        rt_vals = _parse_values_block(rt_block)
                    else:
                        # 如果没有写，就用和 cell_rise 同形状的 0 表
                        rt_vals = np.zeros_like(cr_vals)

                    # ---- 解析 fall_transition（下降边的输出 slew）----
                    mft = re.search(r'fall_transition\s*(\([^{}]*\))?\s*\{', tblock)
                    if mft:
                        ft_block, _ = _brace_block(tblock, mft.end() - 1)
                        ft_vals = _parse_values_block(ft_block)
                    else:
                        ft_vals = np.zeros_like(cf_vals)

                    # ---- 单位统一：把网格和表值都乘上缩放系数 ----
                    idx1_ps = idx1 * t_scale      # 输入 slew 网格，单位 ps
                    idx2_pf = idx2 * c_scale      # 负载电容网格，单位 pF

                    cell_rise = cr_vals * t_scale
                    cell_fall = cf_vals * t_scale
                    rise_tr = rt_vals * t_scale
                    fall_tr = ft_vals * t_scale

                    # 把这一条 arc 的所有信息装到字典里，加到结果列表
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
                    # 如果这一条 arc 有格式问题（比如缺 index 或 values），就跳过
                    continue

    return results


# ===================== 向后兼容（旧接口） =====================

def parse_inv_arc_A_Y_auto(lib_text: str):
    """
    旧接口兼容：简单用 parse_cell_arcs 找到第一个 INVX1 类型的 arc 返回。
    这样老代码如果之前只支持 INV_X1 A->Y 这一条弧，也还能用。

    新代码里一般不用这个接口了。
    """
    # 只要 INVX1 这一类的 arc
    arcs = parse_cell_arcs(lib_text, target_cell_types=["INVX1"])
    if not arcs:
        raise ValueError("No INVX1-like cell found in lib.")

    a0 = arcs[0]
    # 构造一个老版本的格式
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
