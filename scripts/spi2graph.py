
import re, os
import numpy as np

def _read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _to_m(val, unit):
    if unit is None or unit == "":
        return float(val)
    u = unit.lower()
    if u in ["u"]: return float(val)*1e-6
    if u in ["n"]: return float(val)*1e-9
    if u in ["p"]: return float(val)*1e-12
    if u in ["m"]: return float(val)*1e-3
    if u in ["k"]: return float(val)*1e3
    return float(val)

def parse_transistors_spice(text):
    devs = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("*",";","//","*#")):
            continue
        m = re.match(r'^[Mm](\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$', s, re.I)
        if not m:
            continue
        name = "M"+m.group(1)
        d,g,sr,b,model,rest = m.group(2),m.group(3),m.group(4),m.group(5),m.group(6),m.group(7)
        t = "nmos" if re.search(r'nmos', model, re.I) else ("pmos" if re.search(r'pmos', model, re.I) else model.lower())
        # strip trailing comments starting with $
        rest = rest.split('$',1)[0]
        # tokenize by spaces
        toks = [tok for tok in re.split(r'[,\s]+', rest) if tok]
        params = {}
        for tok in toks:
            if '=' in tok:
                k,v = tok.split('=',1)
                params[k.lower()] = v
        W = None; L = None
        if 'w' in params:
            # split numeric and optional single-letter unit
            mwu = re.match(r'([0-9.eE\-\+]+)([a-zA-Z]?)$', params['w'])
            if mwu:
                W = _to_m(mwu.group(1), mwu.group(2))
        if 'l' in params:
            mlu = re.match(r'([0-9.eE\-\+]+)([a-zA-Z]?)$', params['l'])
            if mlu:
                L = _to_m(mlu.group(1), mlu.group(2))
        devs.append({"name": name, "d": d, "g": g, "s": sr, "b": b, "type": t, "W": W, "L": L})
    return devs

def parse_top_subckt_pins(text, cell_hint_regex=r'INV.*1'):
    subs = []
    for m in re.finditer(r'(?im)^\s*\.subckt\s+([^\s]+)\s+(.*)$', text):
        name = m.group(1); pins = m.group(2)
        subs.append((name, pins))
    for name, pins in subs:
        if re.match(cell_hint_regex, name, re.I):
            return name, [p for p in pins.strip().split() if p]
    if subs:
        name, pins = subs[-1]
        return name, [p for p in pins.strip().split() if p]
    return None, []

def extract_wl_features(devs):
    wn = sum(d["W"] for d in devs if d["type"].startswith("n")) if devs else 0.0
    wp = sum(d["W"] for d in devs if d["type"].startswith("p")) if devs else 0.0
    ratio = (wp/wn) if (wn>0 and wp>0) else 0.0
    return {"wp_sum": wp, "wn_sum": wn, "wp_over_wn": ratio}
