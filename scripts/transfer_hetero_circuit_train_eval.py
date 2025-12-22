# -*- coding: utf-8 -*-
"""
transfer_hetero_circuit_train_eval.py

Goal:
- Build heterogeneous circuit graphs from SPICE (devices/nets).
- Pretrain on source lib (e.g. Nangate45) labeled delay.
- Finetune on target lib (e.g. ASAP7) labeled delay with a separate head.
- Optional CORAL alignment on graph embeddings for domain adaptation.
- Robust SPICE resolver:
  * scans spice_root for files
  * fuzzy matches cell_type <-> file/subckt
  * supports explicit cellmap_json
  * outputs unresolved_report.json

Dependencies:
  pip install torch torch_geometric pandas numpy scikit-learn tqdm

Example:
  python transfer_hetero_circuit_train_eval.py \
    --data_dir output \
    --spice_root C:\\Users\\xxx\\CellTiming\\data\\spi \
    --src_lib Nangate45 --tgt_lib ASAP7 \
    --asap7_libfile C:\\...\\ASAP7\\asap7sc6t_26_L_211010.sp \
    --cellmap_json output\\cellmap_auto.json \
    --pretrain_epochs 10 --finetune_epochs 40 \
    --batch_size 64 --lr 3e-3 --y_norm log_standard \
    --mix_src_ratio 0.1 --w_coral 0.05
"""

from __future__ import annotations

import os
import re
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, global_mean_pool


# --------------------------
# Utils
# --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_cell_name(s: str) -> str:
    """
    Normalize cell names for fuzzy matching:
    - uppercase
    - remove non-alnum
    - normalize INVX1 vs INV_X1 etc.
    """
    s = str(s).strip()
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "", s)
    # common normalizations:
    s = s.replace("INVX", "INVX")
    s = s.replace("NANDX", "NANDX")
    s = s.replace("NORX", "NORX")
    s = s.replace("XORX", "XORX")
    return s


def tokenize_name(s: str) -> List[str]:
    # split by transitions and digits
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    chunks = s.split()
    toks = []
    for c in chunks:
        toks += re.findall(r"[A-Z]+|\d+", c)
    return toks


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (len(sa | sb) + 1e-9)


def percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


def print_stats(name: str, y: np.ndarray):
    y = np.asarray(y).reshape(-1)
    msg = (
        f"[{name} stats] n={len(y)} "
        f"min={y.min():.6g} p01={percentile(y,1):.6g} median={np.median(y):.6g} "
        f"p99={percentile(y,99):.6g} max={y.max():.6g} mean={y.mean():.6g} std={y.std():.6g}"
    )
    print(msg)


# --------------------------
# SPICE parsing
# --------------------------

def _to_m(val: str, unit: Optional[str]) -> float:
    if unit is None or unit == "":
        return float(val)
    u = unit.lower()
    if u in ["u"]:
        return float(val) * 1e-6
    if u in ["n"]:
        return float(val) * 1e-9
    if u in ["p"]:
        return float(val) * 1e-12
    if u in ["m"]:
        return float(val) * 1e-3
    if u in ["k"]:
        return float(val) * 1e3
    return float(val)


def parse_transistors_spice(text: str) -> List[Dict[str, Any]]:
    """
    Parse MOS lines like:
      Mxxx D G S B MODEL W=... L=...
    """
    devs = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("*", ";", "//", "*#")):
            continue

        # continuation lines not supported (simple)
        m = re.match(
            r"^[Mm](\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$",
            s,
            re.I,
        )
        if not m:
            continue

        name = "M" + m.group(1)
        d, g, sr, b, model, rest = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)

        t = "nmos" if re.search(r"nmos", model, re.I) else ("pmos" if re.search(r"pmos", model, re.I) else model.lower())

        rest = rest.split("$", 1)[0]
        toks = [tok for tok in re.split(r"[, \t]+", rest) if tok]
        params = {}
        for tok in toks:
            if "=" in tok:
                k, v = tok.split("=", 1)
                params[k.lower()] = v

        W = None
        L = None
        if "w" in params:
            mwu = re.match(r"([0-9.eE\-\+]+)([a-zA-Z]?)$", params["w"])
            if mwu:
                W = _to_m(mwu.group(1), mwu.group(2))
        if "l" in params:
            mlu = re.match(r"([0-9.eE\-\+]+)([a-zA-Z]?)$", params["l"])
            if mlu:
                L = _to_m(mlu.group(1), mlu.group(2))

        devs.append({"name": name, "d": d, "g": g, "s": sr, "b": b, "type": t, "W": W, "L": L})
    return devs


def parse_all_subckts(text: str) -> Dict[str, List[str]]:
    """
    Return subckt_name -> pins list
    """
    subckts = {}
    for m in re.finditer(r"(?im)^\s*\.subckt\s+([^\s]+)\s+(.*)$", text):
        name = m.group(1)
        pins = [p for p in m.group(2).strip().split() if p]
        subckts[name] = pins
    return subckts


def slice_subckt_block(text: str, subckt_name: str) -> Optional[str]:
    """
    Extract .subckt ... .ends block for a given subckt name (exact match).
    """
    # careful with regex - match exact name after .subckt
    pat = re.compile(rf"(?ims)^\s*\.subckt\s+{re.escape(subckt_name)}\b.*?^\s*\.ends\b.*?$")
    m = pat.search(text)
    if not m:
        return None
    return m.group(0)


# --------------------------
# Resolver (files + mapping)
# --------------------------

@dataclass
class ResolverConfig:
    spice_root: str
    src_lib: str = "Nangate45"
    tgt_lib: str = "ASAP7"
    asap7_libfile: str = ""          # path to big .sp with many subckt
    cellmap_json: Optional[str] = None
    strict_cellmap: bool = False     # if true: fail when mapping needed but missing


class SpiceResolver:
    def __init__(self, cfg: ResolverConfig):
        self.cfg = cfg

        # Optional explicit mapping: dataset cell_type -> lib cell name
        self.cellmap: Dict[str, str] = {}
        if cfg.cellmap_json and os.path.exists(cfg.cellmap_json):
            with open(cfg.cellmap_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # allow both normalized or raw keys
            for k, v in raw.items():
                self.cellmap[normalize_cell_name(k)] = v

        # Scan Nangate45 (and other "file-based" libs)
        self.file_index: Dict[str, List[str]] = {}  # lib -> list of file paths
        self.name_index: Dict[str, Dict[str, str]] = {}  # lib -> norm_name -> filepath
        self._scan_lib_files(cfg.src_lib)
        if cfg.tgt_lib != cfg.src_lib:
            self._scan_lib_files(cfg.tgt_lib)

        # Load ASAP7 libfile subckts lazily
        self._asap_text: Optional[str] = None
        self._asap_subckts: Optional[Dict[str, List[str]]] = None
        self._asap_norm_names: Optional[Dict[str, str]] = None  # norm->real

    def _scan_lib_files(self, lib: str):
        lib_dir = os.path.join(self.cfg.spice_root, lib)
        paths = []
        if os.path.isdir(lib_dir):
            for root, _, files in os.walk(lib_dir):
                for fn in files:
                    if fn.lower().endswith((".spi", ".sp")):
                        paths.append(os.path.join(root, fn))
        self.file_index[lib] = paths

        nm = {}
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            nb = normalize_cell_name(base)
            nm[nb] = p
        self.name_index[lib] = nm

    def _ensure_asap_loaded(self):
        if self._asap_text is not None:
            return
        if not self.cfg.asap7_libfile:
            raise FileNotFoundError("asap7_libfile is required for ASAP7.")
        if not os.path.exists(self.cfg.asap7_libfile):
            raise FileNotFoundError(f"asap7_libfile not found: {self.cfg.asap7_libfile}")

        self._asap_text = read_text(self.cfg.asap7_libfile)
        self._asap_subckts = parse_all_subckts(self._asap_text)

        norm_map = {}
        for real in self._asap_subckts.keys():
            norm_map[normalize_cell_name(real)] = real
        self._asap_norm_names = norm_map

    def _fuzzy_match(self, query: str, candidates: List[str], topk: int = 10) -> List[Tuple[str, float]]:
        qn = normalize_cell_name(query)
        qt = tokenize_name(query)
        scored = []
        for c in candidates:
            cn = normalize_cell_name(c)
            if cn == qn:
                scored.append((c, 10.0))
                continue
            # contains bonus
            bonus = 0.0
            if qn in cn or cn in qn:
                bonus += 1.0
            # token overlap
            ct = tokenize_name(c)
            sim = jaccard(qt, ct)
            scored.append((c, sim + bonus))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def dump_cellmap_suggestions(self, dataset_cell_types: List[str], out_json: str):
        """
        Write mapping suggestions for ASAP7 using fuzzy match to subckt names.
        """
        self._ensure_asap_loaded()
        assert self._asap_subckts is not None

        uniq = sorted(set(normalize_cell_name(x) for x in dataset_cell_types))
        asap_names = list(self._asap_subckts.keys())
        res = {}
        for ct_norm in uniq:
            # expand by raw (best-effort)
            top = self._fuzzy_match(ct_norm, asap_names, topk=10)
            # choose best candidate
            best = top[0][0] if top else ""
            res[ct_norm] = best
        safe_mkdir(os.path.dirname(out_json) or ".")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[Dump] wrote cellmap -> {out_json}")

    def resolve_cell_name(self, cell_type: str, lib: str) -> str:
        """
        Return resolved "library cell name" for given dataset cell_type.
        For file-based libs (Nangate45), it's usually file basename.
        For ASAP7, it's subckt name.
        """
        ct_norm = normalize_cell_name(cell_type)

        # explicit mapping
        if ct_norm in self.cellmap:
            return self.cellmap[ct_norm]

        if self.cfg.strict_cellmap and lib.upper() == "ASAP7":
            raise RuntimeError(f"Missing cellmap for {cell_type} while --strict_cellmap enabled.")

        # if not strict, we'll fuzzy match
        return cell_type

    def load_devs_and_pins(self, cell_type: str, lib: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        libU = lib.upper()
        if libU == "ASAP7":
            self._ensure_asap_loaded()
            assert self._asap_text is not None and self._asap_subckts is not None

            resolved = self.resolve_cell_name(cell_type, lib="ASAP7")
            # If resolved is not an actual subckt, fuzzy match against subckts
            if resolved not in self._asap_subckts:
                top = self._fuzzy_match(resolved, list(self._asap_subckts.keys()), topk=10)
                if top:
                    # pick best
                    resolved = top[0][0]
                else:
                    raise RuntimeError(f"ASAP7 subckt not found for cell_type={cell_type} in {self.cfg.asap7_libfile}")

            block = slice_subckt_block(self._asap_text, resolved)
            if block is None:
                raise RuntimeError(f"ASAP7 subckt block not found for {resolved}")

            devs = parse_transistors_spice(block)
            pins = self._asap_subckts.get(resolved, [])

            if len(devs) == 0:
                raise RuntimeError(f"No transistors parsed for cell_type={cell_type} (lib=ASAP7, resolved={resolved}).")
            return devs, pins

        # File-based lib: find a spice file
        cand_paths = self.file_index.get(lib, [])
        if not cand_paths:
            raise FileNotFoundError(f"spice_root has no .sp/.spi for lib={lib}: {os.path.join(self.cfg.spice_root, lib)}")

        # resolve via explicit mapping (optional)
        resolved = self.resolve_cell_name(cell_type, lib=lib)
        rnorm = normalize_cell_name(resolved)

        # direct hit by basename index
        if rnorm in self.name_index.get(lib, {}):
            p = self.name_index[lib][rnorm]
            text = read_text(p)
            devs = parse_transistors_spice(text)
            pins = []
            if len(devs) == 0:
                raise RuntimeError(f"No transistors parsed for cell_type={cell_type} (lib={lib}, file={p}).")
            return devs, pins

        # fuzzy match by basename across files
        basenames = [os.path.splitext(os.path.basename(p))[0] for p in cand_paths]
        top = self._fuzzy_match(resolved, basenames, topk=10)
        if not top:
            tried = [
                os.path.join(self.cfg.spice_root, lib, f"{resolved}_lpe.spi"),
                os.path.join(self.cfg.spice_root, lib, f"{resolved}.spi"),
                os.path.join(self.cfg.spice_root, lib, f"{resolved}.sp"),
            ]
            raise FileNotFoundError(f"{lib} spice file not found for cell_type={cell_type}. Tried: {tried[:3]} ...")

        best_base = top[0][0]
        # choose file whose basename matches best_base
        best_path = None
        for p in cand_paths:
            if os.path.splitext(os.path.basename(p))[0] == best_base:
                best_path = p
                break
        if best_path is None:
            # fallback: first file containing best_base
            for p in cand_paths:
                if best_base in os.path.basename(p):
                    best_path = p
                    break

        if best_path is None:
            raise FileNotFoundError(f"{lib} fuzzy matched {best_base} but could not locate file path.")

        text = read_text(best_path)
        devs = parse_transistors_spice(text)
        if len(devs) == 0:
            raise RuntimeError(f"No transistors parsed for cell_type={cell_type} (lib={lib}, best_file={best_path}).")
        return devs, []


# --------------------------
# Graph building
# --------------------------

def build_hetero_graph(
    row: pd.Series,
    devs: List[Dict[str, Any]],
    y_value: Optional[float],
    vocab: Dict[str, Dict[str, int]],
    num_cols: List[str],
    lib_id: int,
) -> HeteroData:
    """
    Hetero graph with node types:
      - 'dev': transistor devices
      - 'net': nets
    Relations:
      ('dev','conn','net') and ('net','rev','dev')
    Node features:
      dev.x: [W, L, is_n, is_p] (with None->0)
      net.x: learned embedding from net_id (we store net_id)
    Graph-level numeric features from CSV row stored as data['graph'].x_num
    """
    data = HeteroData()

    net_to_id: Dict[str, int] = {}
    def get_net_id(n: str) -> int:
        if n not in net_to_id:
            net_to_id[n] = len(net_to_id)
        return net_to_id[n]

    # Build edges by connecting dev to its terminals
    dev_x = []
    dev_pol = []
    edges_src = []
    edges_dst = []

    for i, d in enumerate(devs):
        W = d.get("W", None)
        L = d.get("L", None)
        W = float(W) if W is not None and not (isinstance(W, float) and math.isnan(W)) else 0.0
        L = float(L) if L is not None and not (isinstance(L, float) and math.isnan(L)) else 0.0
        typ = str(d.get("type", "")).lower()
        is_n = 1.0 if typ.startswith("n") else 0.0
        is_p = 1.0 if typ.startswith("p") else 0.0
        dev_x.append([W, L, is_n, is_p])

        for term in [d["d"], d["g"], d["s"], d["b"]]:
            nid = get_net_id(term)
            edges_src.append(i)
            edges_dst.append(nid)

    data["dev"].x = torch.tensor(dev_x, dtype=torch.float32)
    data["dev"].num_nodes = data["dev"].x.size(0)

    net_ids = list(range(len(net_to_id)))
    data["net"].net_id = torch.tensor(net_ids, dtype=torch.long)
    data["net"].num_nodes = len(net_ids)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    data["dev", "conn", "net"].edge_index = edge_index
    data["net", "rev", "dev"].edge_index = torch.tensor([edges_dst, edges_src], dtype=torch.long)

    # graph-level numeric features from row
    x_num = row[num_cols].astype(np.float32).values
    data["graph"].x_num = torch.tensor(x_num, dtype=torch.float32).view(1, -1)
    data["graph"].lib_id = torch.tensor([lib_id], dtype=torch.long)

    # optional y
    if y_value is not None:
        data["graph"].y = torch.tensor([float(y_value)], dtype=torch.float32)

    # categorical features also on graph
    ct = str(row.get("cell_type", "UNK"))
    pol = str(row.get("pol", "UNK"))
    data["graph"].cell_type_id = torch.tensor([vocab["cell_type"].get(ct, 0)], dtype=torch.long)
    data["graph"].pol_id = torch.tensor([vocab["pol"].get(pol, 0)], dtype=torch.long)

    return data


def build_vocab(df_all: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    def make_map(values: List[str]) -> Dict[str, int]:
        uniq = sorted(set(values))
        m = {"UNK": 0}
        for i, v in enumerate(uniq):
            if v == "UNK":
                continue
            m[v] = len(m)
        return m

    cell_map = make_map(df_all["cell_type"].astype(str).tolist())
    pol_map = make_map(df_all.get("pol", pd.Series(["UNK"] * len(df_all))).astype(str).tolist())
    print(f"[Vocab] cell_types={len(cell_map)} pols={len(pol_map)}")
    return {"cell_type": cell_map, "pol": pol_map}


# --------------------------
# Model
# --------------------------

class HeteroCircuitEncoder(nn.Module):
    def __init__(self, net_vocab_size: int = 1024, net_emb_dim: int = 16, hidden: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.net_emb = nn.Embedding(net_vocab_size, net_emb_dim)

        # dev input: 4 dims; net input: net_emb_dim
        self.dev_lin = Linear(4, hidden)
        self.net_lin = Linear(net_emb_dim, hidden)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv(
                {
                    ("dev", "conn", "net"): GATv2Conv((-1, -1), hidden // heads, heads=heads, add_self_loops=False),
                    ("net", "rev", "dev"): GATv2Conv((-1, -1), hidden // heads, heads=heads, add_self_loops=False),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.out_lin_dev = Linear(hidden, hidden)
        self.out_lin_net = Linear(hidden, hidden)

    def forward(self, data: HeteroData) -> torch.Tensor:
        # node init
        net_id = data["net"].net_id.clamp(min=0)
        net_id = net_id % self.net_emb.num_embeddings
        x_net = self.net_emb(net_id)

        x_dict = {
            "dev": F.relu(self.dev_lin(data["dev"].x)),
            "net": F.relu(self.net_lin(x_net)),
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # pool dev nodes as graph embedding
        # Need batch vector: if absent, treat as single graph
        if hasattr(data["dev"], "batch") and data["dev"].batch is not None:
            batch = data["dev"].batch
        else:
            batch = torch.zeros(data["dev"].num_nodes, dtype=torch.long, device=data["dev"].x.device)

        g = global_mean_pool(self.out_lin_dev(x_dict["dev"]), batch)
        return g  # [B, hidden]


class TransferRegressor(nn.Module):
    def __init__(self, num_graph_num: int, num_cell_types: int, num_pols: int, hidden: int = 64, emb_dim: int = 16, head_hidden: int = 128):
        super().__init__()
        self.encoder = HeteroCircuitEncoder(hidden=hidden)

        self.ct_emb = nn.Embedding(max(2, num_cell_types), emb_dim)
        self.pol_emb = nn.Embedding(max(2, num_pols), emb_dim)

        in_dim = hidden + num_graph_num + emb_dim * 2

        def make_head():
            return nn.Sequential(
                nn.Linear(in_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.ReLU(),
                nn.Linear(head_hidden // 2, 1),
            )

        self.head_src = make_head()
        self.head_tgt = make_head()

    def forward(self, data: HeteroData, domain: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pred, embedding
        """
        g = self.encoder(data)  # [B, hidden]
        x_num = data["graph"].x_num
        ct = data["graph"].cell_type_id
        pol = data["graph"].pol_id
        z = torch.cat([g, x_num, self.ct_emb(ct).view(ct.size(0), -1), self.pol_emb(pol).view(pol.size(0), -1)], dim=-1)
        if domain == "src":
            return self.head_src(z).view(-1), g
        else:
            return self.head_tgt(z).view(-1), g


def coral_loss(xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """
    CORAL: align covariance of embeddings.
    xs: [Bs, d], xt: [Bt, d]
    """
    if xs.size(0) < 2 or xt.size(0) < 2:
        return torch.tensor(0.0, device=xs.device)
    xs = xs - xs.mean(dim=0, keepdim=True)
    xt = xt - xt.mean(dim=0, keepdim=True)
    cs = (xs.t() @ xs) / (xs.size(0) - 1)
    ct = (xt.t() @ xt) / (xt.size(0) - 1)
    return ((cs - ct) ** 2).mean()


# --------------------------
# Train / Eval
# --------------------------

@torch.no_grad()
def eval_loader(model: TransferRegressor, loader: DataLoader, device: torch.device, domain: str, y_inv=None) -> Dict[str, float]:
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch, domain=domain)
        y = batch["graph"].y.view(-1)
        if y_inv is not None:
            pred_np = y_inv(pred.detach().cpu().numpy())
            y_np = y_inv(y.detach().cpu().numpy())
            ps.append(pred_np)
            ys.append(y_np)
        else:
            ps.append(pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    if len(y) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": float(r2)}


def train_pretrain(
    model: TransferRegressor,
    loader_src: DataLoader,
    loader_src_test: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    y_inv_src=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_rmse = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader_src:
            batch = batch.to(device)
            pred, _ = model(batch, domain="src")
            y = batch["graph"].y.view(-1)
            loss = F.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        metrics = eval_loader(model, loader_src_test, device, domain="src", y_inv=y_inv_src)
        print(f"[Pretrain {ep:03d}] loss={np.mean(losses):.6f} SRC_TEST={metrics}")
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]


def train_finetune(
    model: TransferRegressor,
    loader_tgt_train: DataLoader,
    loader_tgt_val: DataLoader,
    loader_src_mix: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    y_inv_tgt=None,
    mix_src_ratio: float = 0.1,
    w_src: float = 0.1,
    w_tgt: float = 1.0,
    w_coral: float = 0.0,
    freeze_encoder: bool = False,
    out_dir: str = "output",
):
    safe_mkdir(out_dir)
    # optionally freeze encoder
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        src_iter = iter(loader_src_mix) if loader_src_mix is not None else None

        for tgt_batch in loader_tgt_train:
            tgt_batch = tgt_batch.to(device)
            pred_t, emb_t = model(tgt_batch, domain="tgt")
            y_t = tgt_batch["graph"].y.view(-1)
            Lt = F.smooth_l1_loss(pred_t, y_t)

            Ls = torch.tensor(0.0, device=device)
            emb_s = None
            if src_iter is not None and mix_src_ratio > 0:
                # sample src occasionally
                if random.random() < mix_src_ratio:
                    try:
                        src_batch = next(src_iter)
                    except StopIteration:
                        src_iter = iter(loader_src_mix)
                        src_batch = next(src_iter)
                    src_batch = src_batch.to(device)
                    pred_s, emb_s = model(src_batch, domain="src")
                    y_s = src_batch["graph"].y.view(-1)
                    Ls = F.smooth_l1_loss(pred_s, y_s)

            Lc = torch.tensor(0.0, device=device)
            if w_coral > 0 and emb_s is not None:
                Lc = coral_loss(emb_s, emb_t)

            loss = w_tgt * Lt + w_src * Ls + w_coral * Lc

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.item()))

        metrics = eval_loader(model, loader_tgt_val, device, domain="tgt", y_inv=y_inv_tgt)
        print(f"[Finetune {ep:03d}] train_loss={np.mean(losses):.6f} TGT_VAL={metrics}")

        if metrics["RMSE"] < best:
            best = metrics["RMSE"]
            ckpt = os.path.join(out_dir, "best_transfer_targetval.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  [*] saved best -> {ckpt} (tgt_val_RMSE={best:.6f})")


# --------------------------
# y normalization
# --------------------------

class YNorm:
    """
    Supports:
      - none
      - standard (z-score)
      - log_standard: y' = log1p(y), then standard
    For finetune we recommend target has its own scaler to avoid scale mismatch.
    """
    def __init__(self, mode: str):
        self.mode = mode
        self.scaler = StandardScaler()

    def fit(self, y: np.ndarray):
        y = y.reshape(-1, 1).astype(np.float32)
        if self.mode == "log_standard":
            y = np.log1p(np.clip(y, a_min=0.0, a_max=None))
        if self.mode == "standard" or self.mode == "log_standard":
            self.scaler.fit(y)
        return self

    def fwd(self, y: np.ndarray) -> np.ndarray:
        y = y.reshape(-1, 1).astype(np.float32)
        if self.mode == "none":
            return y.reshape(-1)
        if self.mode == "log_standard":
            y = np.log1p(np.clip(y, a_min=0.0, a_max=None))
        y = self.scaler.transform(y)
        return y.reshape(-1)

    def inv(self, y: np.ndarray) -> np.ndarray:
        y = y.reshape(-1, 1).astype(np.float32)
        if self.mode == "none":
            return y.reshape(-1)
        y = self.scaler.inverse_transform(y)
        if self.mode == "log_standard":
            y = np.expm1(y)
        return y.reshape(-1)


# --------------------------
# Main
# --------------------------

def load_csvs(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expect:
      - src_delay.csv
      - tgt_delay_labeled.csv
      - tgt_delay_unlabeled.csv (optional)
      - meta.json (optional)
    """
    src_path = os.path.join(data_dir, "src_delay.csv")
    tgt_lab_path = os.path.join(data_dir, "tgt_delay_labeled.csv")
    tgt_unlab_path = os.path.join(data_dir, "tgt_delay_unlabeled.csv")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"missing {src_path}")
    if not os.path.exists(tgt_lab_path):
        raise FileNotFoundError(f"missing {tgt_lab_path}")

    src_df = pd.read_csv(src_path)
    tgt_lab = pd.read_csv(tgt_lab_path)
    tgt_unlab = pd.read_csv(tgt_unlab_path) if os.path.exists(tgt_unlab_path) else pd.DataFrame()

    return src_df, tgt_lab, tgt_unlab


def pick_numeric_columns(df: pd.DataFrame) -> List[str]:
    # exclude obvious non-numeric columns
    ban = {"cell_type", "pol", "lib", "spice", "spice_path", "netlist", "delay", "y"}
    cols = []
    for c in df.columns:
        if c in ban:
            continue
        if df[c].dtype == object:
            # try coercion
            try:
                pd.to_numeric(df[c].iloc[:50], errors="raise")
            except Exception:
                continue
        cols.append(c)
    # ensure stable order
    cols = sorted(cols)
    return cols


def build_graphs_for_df(
    df: pd.DataFrame,
    resolver: SpiceResolver,
    lib: str,
    vocab: Dict[str, Dict[str, int]],
    num_cols: List[str],
    y_col: str,
    y_fwd_fn,
    lib_id: int,
    strict: bool,
    tag: str,
    out_dir: str,
) -> Tuple[List[HeteroData], Dict[str, Any]]:
    graphs = []
    report = {"tag": tag, "lib": lib, "built": 0, "skipped": 0, "errors": []}

    it = tqdm(range(len(df)), desc=f"Graphs[{lib}] {tag}")
    for i in it:
        row = df.iloc[i]
        ct = str(row.get("cell_type", "UNK"))
        try:
            devs, pins = resolver.load_devs_and_pins(ct, lib=lib)
            yv = None
            if y_col in df.columns:
                y_raw = float(row[y_col])
                yv = float(y_fwd_fn(np.array([y_raw], dtype=np.float32))[0])
            g = build_hetero_graph(row, devs, yv, vocab, num_cols, lib_id=lib_id)
            graphs.append(g)
            report["built"] += 1
        except Exception as e:
            report["skipped"] += 1
            report["errors"].append({"idx": int(i), "cell_type": ct, "err": str(e)})
            continue

    keep_ratio = report["built"] / max(1, report["built"] + report["skipped"])
    print(f"[Graphs[{lib}]] built={report['built']}/{len(df)} skipped={report['skipped']} keep_ratio={keep_ratio:.3f}")
    # write report for debugging
    safe_mkdir(out_dir)
    with open(os.path.join(out_dir, f"unresolved_{lib}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if strict and keep_ratio < 0.95:
        raise RuntimeError(f"Too many graphs unresolved for lib={lib}: keep_ratio={keep_ratio:.3f} < 0.95. Fix spice_root/cellmap/spice parsing. See unresolved_{lib}_{tag}.json")

    return graphs, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="output")
    ap.add_argument("--spice_root", type=str, required=True, help="root dir containing subfolders Nangate45/, ASAP7/, ...")
    ap.add_argument("--src_lib", type=str, default="Nangate45")
    ap.add_argument("--tgt_lib", type=str, default="ASAP7")
    ap.add_argument("--asap7_libfile", type=str, default="", help="Required if tgt_lib==ASAP7 or you want dump_cellmap")
    ap.add_argument("--cellmap_json", type=str, default=None)
    ap.add_argument("--strict_cellmap", action="store_true")
    ap.add_argument("--dump_cellmap", type=str, default=None, help="If set, dump auto suggestions mapping for dataset cell_type -> ASAP7 subckt")
    ap.add_argument("--y_col", type=str, default="delay")
    ap.add_argument("--y_norm", type=str, default="log_standard", choices=["none","standard","log_standard"])
    ap.add_argument("--tgt_val_ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pretrain_epochs", type=int, default=10)
    ap.add_argument("--finetune_epochs", type=int, default=40)

    ap.add_argument("--mix_src_ratio", type=float, default=0.10)
    ap.add_argument("--w_src", type=float, default=0.10)
    ap.add_argument("--w_tgt", type=float, default=1.00)
    ap.add_argument("--w_coral", type=float, default=0.00)

    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[Device] {device}")

    src_df, tgt_lab_df, tgt_unlab_df = load_csvs(args.data_dir)

    # Basic stats
    print_stats("SRC delay", src_df[args.y_col].values.astype(np.float32))
    print_stats("TGT_LABELED delay", tgt_lab_df[args.y_col].values.astype(np.float32))

    # Dump cellmap suggestions if needed
    cfg = ResolverConfig(
        spice_root=args.spice_root,
        src_lib=args.src_lib,
        tgt_lib=args.tgt_lib,
        asap7_libfile=args.asap7_libfile,
        cellmap_json=args.cellmap_json,
        strict_cellmap=args.strict_cellmap,
    )
    resolver = SpiceResolver(cfg)

    if args.dump_cellmap:
        # Use both src and tgt cell types to suggest mapping
        all_ct = list(src_df["cell_type"].astype(str).values) + list(tgt_lab_df["cell_type"].astype(str).values)
        resolver.dump_cellmap_suggestions(all_ct, args.dump_cellmap)
        # continue running training still okay

    # Split src: train/test
    src_train_df, src_test_df = train_test_split(src_df, test_size=0.15, random_state=args.seed, shuffle=True)

    # Split tgt labeled: train/val
    tgt_train_df, tgt_val_df = train_test_split(tgt_lab_df, test_size=args.tgt_val_ratio, random_state=args.seed, shuffle=True)

    print(f"[Split] src_train={len(src_train_df)} src_test={len(src_test_df)} | tgt_train={len(tgt_train_df)} tgt_val={len(tgt_val_df)}")

    # choose numeric columns intersection to avoid mismatch
    num_cols_src = pick_numeric_columns(src_df)
    num_cols_tgt = pick_numeric_columns(tgt_lab_df)
    num_cols = sorted(set(num_cols_src) & set(num_cols_tgt))
    if len(num_cols) == 0:
        raise RuntimeError("No numeric feature columns found. Ensure CSV has numeric features besides delay/cell_type/pol.")

    print(f"[Feat] numeric_dim={len(num_cols)}")

    # y normalization:
    ysrc = src_df[args.y_col].values.astype(np.float32)
    ytgt = tgt_lab_df[args.y_col].values.astype(np.float32)

    yn_src = YNorm(args.y_norm).fit(ysrc)
    yn_tgt = YNorm(args.y_norm).fit(ytgt)

    print_stats("SRC (normalized)", yn_src.fwd(ysrc))
    print_stats("TGT (normalized, tgt-scaler)", yn_tgt.fwd(ytgt))

    # vocab based on concatenated dfs
    df_all = pd.concat([src_df, tgt_lab_df], axis=0, ignore_index=True)
    if "pol" not in df_all.columns:
        df_all["pol"] = "UNK"
        src_train_df = src_train_df.copy()
        src_test_df = src_test_df.copy()
        tgt_train_df = tgt_train_df.copy()
        tgt_val_df = tgt_val_df.copy()
        for d in [src_train_df, src_test_df, tgt_train_df, tgt_val_df]:
            if "pol" not in d.columns:
                d["pol"] = "UNK"

    vocab = build_vocab(df_all)

    # Build graphs (strict on src, but you can turn off by not using --strict_cellmap)
    g_src_train, _ = build_graphs_for_df(
        src_train_df, resolver, args.src_lib, vocab, num_cols, args.y_col, yn_src.fwd,
        lib_id=0, strict=True, tag="src_train", out_dir=args.out_dir
    )
    g_src_test, _ = build_graphs_for_df(
        src_test_df, resolver, args.src_lib, vocab, num_cols, args.y_col, yn_src.fwd,
        lib_id=0, strict=False, tag="src_test", out_dir=args.out_dir
    )
    g_tgt_train, _ = build_graphs_for_df(
        tgt_train_df, resolver, args.tgt_lib, vocab, num_cols, args.y_col, yn_tgt.fwd,
        lib_id=1, strict=False, tag="tgt_train", out_dir=args.out_dir
    )
    g_tgt_val, _ = build_graphs_for_df(
        tgt_val_df, resolver, args.tgt_lib, vocab, num_cols, args.y_col, yn_tgt.fwd,
        lib_id=1, strict=False, tag="tgt_val", out_dir=args.out_dir
    )

    # loaders
    loader_src = DataLoader(g_src_train, batch_size=args.batch_size, shuffle=True)
    loader_src_test = DataLoader(g_src_test, batch_size=args.batch_size, shuffle=False)

    # for finetune, tgt is small; keep shuffle
    loader_tgt_train = DataLoader(g_tgt_train, batch_size=args.batch_size, shuffle=True)
    loader_tgt_val = DataLoader(g_tgt_val, batch_size=args.batch_size, shuffle=False)

    loader_src_mix = DataLoader(g_src_train, batch_size=args.batch_size, shuffle=True) if args.mix_src_ratio > 0 else None

    # model
    model = TransferRegressor(
        num_graph_num=len(num_cols),
        num_cell_types=len(vocab["cell_type"]),
        num_pols=len(vocab["pol"]),
        hidden=64,
        emb_dim=16,
        head_hidden=128,
    ).to(device)

    # Pretrain
    if args.pretrain_epochs > 0:
        train_pretrain(model, loader_src, loader_src_test, device, epochs=args.pretrain_epochs, lr=args.lr, y_inv_src=yn_src.inv)

    # Finetune
    if args.finetune_epochs > 0:
        train_finetune(
            model,
            loader_tgt_train,
            loader_tgt_val,
            loader_src_mix,
            device,
            epochs=args.finetune_epochs,
            lr=args.lr * (0.3 if args.freeze_encoder else 1.0),
            y_inv_tgt=yn_tgt.inv,
            mix_src_ratio=args.mix_src_ratio,
            w_src=args.w_src,
            w_tgt=args.w_tgt,
            w_coral=args.w_coral,
            freeze_encoder=args.freeze_encoder,
            out_dir=args.out_dir,
        )

    # Final report
    # Load best if exists
    best_path = os.path.join(args.out_dir, "best_transfer_targetval.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    src_test_metrics = eval_loader(model, loader_src_test, device, domain="src", y_inv=yn_src.inv)
    tgt_val_metrics = eval_loader(model, loader_tgt_val, device, domain="tgt", y_inv=yn_tgt.inv)
    print(f"[SRC TEST] {src_test_metrics}")
    print(f"[TGT VAL] {tgt_val_metrics}")
    print("[Done]")


if __name__ == "__main__":
    main()



