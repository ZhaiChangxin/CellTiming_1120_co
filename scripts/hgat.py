
import torch
import torch.nn as nn
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
try:
    import dgl
    from dgl.nn import HeteroGraphConv, GATConv
except Exception as e:
    raise ImportError("DGL is required for HGAT. Please install dgl (CPU/GPU).")

class HGATDesignEncoder(nn.Module):
    def __init__(self, in_dim_map, hid=64, out=64, num_heads=1):
        super().__init__()
        rels = ["gate_of", "sd_to", "back_sd"]
        self.embed = nn.ModuleDict({nt: nn.Linear(in_dim_map[nt], hid) for nt in in_dim_map})
        self.layer1 = HeteroGraphConv({r: GATConv(hid, hid, num_heads=num_heads) for r in rels}, aggregate='sum')
        self.layer2 = HeteroGraphConv({r: GATConv(hid, hid, num_heads=num_heads) for r in rels}, aggregate='sum')
        self.readout = nn.Sequential(nn.Linear(hid, out), nn.ReLU(), nn.Linear(out, out))

    def forward(self, g, feats):
        h = {nt: self.embed[nt](feats[nt]) for nt in feats}
        h = self.layer1(g, h)
        h = {k: v.mean(1) for k, v in h.items()}
        h = {k: torch.relu(v) for k, v in h.items()}
        h = self.layer2(g, h)
        h = {k: v.mean(1) for k, v in h.items()}
        mos = []
        for nt in ["PMOS","NMOS"]:
            if nt in h and h[nt].shape[0] > 0:
                mos.append(h[nt].mean(dim=0, keepdim=True))
        if len(mos) == 0:
            mos = [v.mean(dim=0, keepdim=True) for v in h.values()]
        z = torch.mean(torch.cat(mos, dim=0), dim=0)
        return self.readout(z)
# hgat.py - 追加以下辅助函数（直接复用 train.py 的逻辑）
def build_dgl_graph_from_devs(devs, top_pins):
    import dgl, torch, numpy as np
    nets = {}
    def net_id(n):
        if n not in nets: nets[n] = len(nets)
        return nets[n]

    p_count=n_count=0
    gate_src_p= []; gate_dst_p= []
    gate_src_n= []; gate_dst_n= []
    sd_src_p=   []; sd_dst_p=   []
    sd_src_n=   []; sd_dst_n=   []

    for d in devs:
        if d["type"].startswith("p"):
            mid = p_count; p_count += 1
            gate_src_p.append(net_id(d["g"])); gate_dst_p.append(mid)
            sd_src_p.extend([mid, mid]); sd_dst_p.extend([net_id(d["s"]), net_id(d["d"])])
        else:
            mid = n_count; n_count += 1
            gate_src_n.append(net_id(d["g"])); gate_dst_n.append(mid)
            sd_src_n.extend([mid, mid]); sd_dst_n.extend([net_id(d["s"]), net_id(d["d"])])

    data_dict = {}
    if p_count > 0:
        data_dict[('NET','gate_of','PMOS')] = (torch.tensor(gate_src_p), torch.tensor(gate_dst_p))
        data_dict[('PMOS','sd_to','NET')]   = (torch.tensor(sd_src_p),  torch.tensor(sd_dst_p))
        data_dict[('NET','back_sd','PMOS')] = (torch.tensor(sd_dst_p),  torch.tensor(sd_src_p))
    if n_count > 0:
        data_dict[('NET','gate_of','NMOS')] = (torch.tensor(gate_src_n), torch.tensor(gate_dst_n))
        data_dict[('NMOS','sd_to','NET')]   = (torch.tensor(sd_src_n),  torch.tensor(sd_dst_n))
        data_dict[('NET','back_sd','NMOS')] = (torch.tensor(sd_dst_n),  torch.tensor(sd_src_n))

    g = dgl.heterograph(data_dict, num_nodes_dict={'NET': len(nets), 'PMOS': p_count, 'NMOS': n_count})

    f_net = []
    for name, nid in sorted(nets.items(), key=lambda x:x[1]):
        is_vdd = 1.0 if name.upper()=="VDD" else 0.0
        is_vss = 1.0 if name.upper()=="VSS" else 0.0
        is_a   = 1.0 if name.upper()=="A" else 0.0
        is_y   = 1.0 if name.upper() in ("Y","ZN") else 0.0
        f_net.append([is_vdd,is_vss,is_a,is_y])
    f_net = torch.tensor(np.array(f_net, dtype=np.float32))

    def mos_feats(list_dev):
        arr = []
        for d in list_dev:
            W = d["W"] if d["W"] is not None else 0.0
            L = d["L"] if d["L"] is not None else 0.0
            arr.append([W, L])
        if len(arr)==0:
            return torch.zeros((0,2), dtype=torch.float32)
        return torch.tensor(np.array(arr, dtype=np.float32))

    f_p = mos_feats([d for d in devs if d["type"].startswith("p")])
    f_n = mos_feats([d for d in devs if d["type"].startswith("n")])

    feats = {'NET': f_net, 'PMOS': f_p, 'NMOS': f_n}
    in_dim_map = {'NET': f_net.shape[1] if f_net.numel() else 4, 'PMOS': 2, 'NMOS': 2}
    return g, feats, in_dim_map

def build_graph_from_spice(spi_path):
    text = open(spi_path,"r",encoding="utf-8",errors="ignore").read()
    devs = parse_transistors_spice(text)
    _, pins = parse_top_subckt_pins(text)
    return build_dgl_graph_from_devs(devs, pins)
