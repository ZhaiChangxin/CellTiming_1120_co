import torch
import torch.nn as nn
from torch.nn import functional as F
from spi2graph import parse_transistors_spice, parse_top_subckt_pins

try:
    import dgl
    from dgl.nn import HeteroGraphConv, GATConv
except Exception as e:
    raise ImportError("DGL is required for HGAT. Please install dgl (CPU/GPU).")


class HGATDesignEncoder(nn.Module):
    """
    Optimized HGAT:
    - multi-head GAT
    - dropout
    - residual + LayerNorm per node type
    - stable readout (mean pooling across PMOS/NMOS, then MLP)
    - L2 normalize output embedding
    """
    def __init__(self, in_dim_map, hid=64, out=64, num_heads=4, dropout=0.15):
        super().__init__()
        rels = ["gate_of", "sd_to", "back_sd"]

        self.embed = nn.ModuleDict({nt: nn.Linear(in_dim_map[nt], hid) for nt in in_dim_map})
        self.dropout = nn.Dropout(dropout)

        # Two hetero-GAT layers
        self.layer1 = HeteroGraphConv(
            {r: GATConv(hid, hid, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)
             for r in rels},
            aggregate="sum"
        )
        self.layer2 = HeteroGraphConv(
            {r: GATConv(hid, hid, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)
             for r in rels},
            aggregate="sum"
        )

        # LayerNorm per node type
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(hid) for nt in in_dim_map})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(hid) for nt in in_dim_map})

        self.readout = nn.Sequential(
            nn.Linear(hid, out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out, out)
        )

    def forward(self, g, feats):
        # input projection
        h = {nt: self.embed[nt](feats[nt]) for nt in feats}
        h = {nt: self.dropout(F.relu(hv)) for nt, hv in h.items()}

        # layer1
        h1 = self.layer1(g, h)
        h1 = {k: v.mean(1) for k, v in h1.items()}  # [N, heads, hid] -> [N, hid]
        h1 = {k: self.dropout(F.relu(self.norm1[k](v))) for k, v in h1.items()}

        # residual (align node types)
        for nt in h1:
            if nt in h and h[nt].shape == h1[nt].shape:
                h1[nt] = h1[nt] + h[nt]

        # layer2
        h2 = self.layer2(g, h1)
        h2 = {k: v.mean(1) for k, v in h2.items()}
        h2 = {k: self.dropout(F.relu(self.norm2[k](v))) for k, v in h2.items()}

        # readout: average PMOS/NMOS embeddings (fallback to all types)
        mos = []
        for nt in ["PMOS", "NMOS"]:
            if nt in h2 and h2[nt].shape[0] > 0:
                mos.append(h2[nt].mean(dim=0, keepdim=True))
        if len(mos) == 0:
            mos = [v.mean(dim=0, keepdim=True) for v in h2.values()]

        z = torch.mean(torch.cat(mos, dim=0), dim=0)  # [hid]
        z = self.readout(z)  # [out]
        z = F.normalize(z, dim=0)
        return z


# ---------- Graph builder (keep your existing logic, compatible) ----------
def build_dgl_graph_from_devs(devs, top_pins):
    import dgl, torch, numpy as np

    nets = {}
    def net_id(n):
        if n not in nets:
            nets[n] = len(nets)
        return nets[n]

    p_count = n_count = 0
    gate_src_p, gate_dst_p = [], []
    gate_src_n, gate_dst_n = [], []
    sd_src_p, sd_dst_p = [], []
    sd_src_n, sd_dst_n = [], []

    for d in devs:
        if d["type"].startswith("p"):
            mid = p_count
            p_count += 1
            gate_src_p.append(net_id(d["g"]))
            gate_dst_p.append(mid)
            sd_src_p.extend([mid, mid])
            sd_dst_p.extend([net_id(d["s"]), net_id(d["d"])])
        else:
            mid = n_count
            n_count += 1
            gate_src_n.append(net_id(d["g"]))
            gate_dst_n.append(mid)
            sd_src_n.extend([mid, mid])
            sd_dst_n.extend([net_id(d["s"]), net_id(d["d"])])

    data_dict = {}
    if p_count > 0:
        data_dict[("NET", "gate_of", "PMOS")] = (torch.tensor(gate_src_p), torch.tensor(gate_dst_p))
        data_dict[("PMOS", "sd_to", "NET")] = (torch.tensor(sd_src_p), torch.tensor(sd_dst_p))
        data_dict[("NET", "back_sd", "PMOS")] = (torch.tensor(sd_dst_p), torch.tensor(sd_src_p))
    if n_count > 0:
        data_dict[("NET", "gate_of", "NMOS")] = (torch.tensor(gate_src_n), torch.tensor(gate_dst_n))
        data_dict[("NMOS", "sd_to", "NET")] = (torch.tensor(sd_src_n), torch.tensor(sd_dst_n))
        data_dict[("NET", "back_sd", "NMOS")] = (torch.tensor(sd_dst_n), torch.tensor(sd_src_n))

    g = dgl.heterograph(data_dict, num_nodes_dict={"NET": len(nets), "PMOS": p_count, "NMOS": n_count})

    f_net = []
    for name, nid in sorted(nets.items(), key=lambda x: x[1]):
        is_vdd = 1.0 if name.upper() == "VDD" else 0.0
        is_vss = 1.0 if name.upper() == "VSS" else 0.0
        is_a = 1.0 if name.upper() == "A" else 0.0
        is_y = 1.0 if name.upper() in ("Y", "ZN") else 0.0
        f_net.append([is_vdd, is_vss, is_a, is_y])
    f_net = torch.tensor(np.array(f_net, dtype=np.float32))

    def mos_feats(list_dev):
        arr = []
        for d in list_dev:
            raw_W = d["W"] if d["W"] is not None else 0.0
            raw_L = d["L"] if d["L"] is not None else 0.0
            # meters -> um
            W = raw_W * 1e6
            L = raw_L * 1e6
            arr.append([W, L])
        if len(arr) == 0:
            return torch.zeros((0, 2), dtype=torch.float32)
        return torch.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats([d for d in devs if d["type"].startswith("p")])
    f_n = mos_feats([d for d in devs if d["type"].startswith("n")])

    feats = {"NET": f_net, "PMOS": f_p, "NMOS": f_n}
    in_dim_map = {"NET": f_net.shape[1] if f_net.numel() else 4, "PMOS": 2, "NMOS": 2}
    return g, feats, in_dim_map


def build_graph_from_spice(spi_path):
    text = open(spi_path, "r", encoding="utf-8", errors="ignore").read()
    devs = parse_transistors_spice(text)
    _, pins = parse_top_subckt_pins(text)
    return build_dgl_graph_from_devs(devs, pins)
