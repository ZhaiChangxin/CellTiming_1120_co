import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


# ==========================================
#   1. 图构建辅助函数 (核心修复在此)
# ==========================================
def build_dgl_graph_from_devs(devs, pins):
    """
    将解析后的 SPICE 器件列表转换为 DGL 异构图。
    """

    # 1. 建立节点映射 (Name -> ID)
    net_map = {}
    pmos_map = {}
    nmos_map = {}

    # [修复逻辑] 确保 pins 先被加入 net_map，保证它们一定存在
    for p in pins:
        if p not in net_map:
            net_map[p] = len(net_map)

    # 收集所有的 Net (从器件连接中)
    for dev in devs:
        for net_name in dev.get('nodes', []):
            if net_name not in net_map:
                net_map[net_name] = len(net_map)

    # 收集器件
    pmos_feats_list = []
    nmos_feats_list = []

    for dev in devs:
        # 获取 W/L 等参数
        # 兼容处理: 有些解析结果可能是 'w': '1.0e-7' 字符串
        try:
            w = float(dev.get('w', 1e-7))
            l = float(dev.get('l', 1e-9))
        except:
            w, l = 1e-7, 1e-9

        feat = [w, l]

        # 区分 PMOS / NMOS
        d_type = dev.get('subtype', 'n').lower()  # 'p' or 'n'

        if 'p' in d_type:  # PMOS
            pmos_map[dev['name']] = len(pmos_map)
            pmos_feats_list.append(feat)
        else:  # NMOS
            nmos_map[dev['name']] = len(nmos_map)
            nmos_feats_list.append(feat)

    # 2. 构建边 (Edges)
    data_dict = {
        ('net', 'to_p', 'pmos'): ([], []),
        ('pmos', 'to_n', 'net'): ([], []),
        ('net', 'to_nm', 'nmos'): ([], []),
        ('nmos', 'to_nm', 'net'): ([], [])
    }

    for dev in devs:
        d_name = dev['name']
        d_type = dev.get('subtype', 'n').lower()

        # 确定器件 ID 和 对应的边类型键值
        if 'p' in d_type:
            if d_name not in pmos_map: continue
            d_id = pmos_map[d_name]
            u_key, v_key = ('net', 'to_p', 'pmos'), ('pmos', 'to_n', 'net')
        else:
            if d_name not in nmos_map: continue
            d_id = nmos_map[d_name]
            u_key, v_key = ('net', 'to_nm', 'nmos'), ('nmos', 'to_nm', 'net')

        # 建立连接: Net <-> Device
        for net_name in dev.get('nodes', []):
            if net_name in net_map:
                n_id = net_map[net_name]
                # Bidirectional connection
                data_dict[u_key][0].append(n_id)
                data_dict[u_key][1].append(d_id)
                data_dict[v_key][0].append(d_id)
                data_dict[v_key][1].append(n_id)

    # 3. 创建 DGL Graph [CRITICAL FIX]
    # 显式告诉 DGL 每种节点有多少个，防止因孤立节点导致推断数量偏少而报错
    num_nodes_dict = {
        'net': len(net_map),
        'pmos': len(pmos_map),
        'nmos': len(nmos_map)
    }

    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    # 4. 填充特征 Tensor
    # Net 特征: [is_pin, 0, 0, 0] (示例，共4维)
    net_feat_dim = 4
    net_feats = torch.zeros((g.num_nodes('net'), net_feat_dim), dtype=torch.float32)

    for pin in pins:
        if pin in net_map:
            net_feats[net_map[pin], 0] = 1.0

    pmos_feats = torch.tensor(pmos_feats_list, dtype=torch.float32) if pmos_feats_list else torch.zeros((0, 2))
    nmos_feats = torch.tensor(nmos_feats_list, dtype=torch.float32) if nmos_feats_list else torch.zeros((0, 2))

    feats = {
        'net': net_feats,
        'pmos': pmos_feats,
        'nmos': nmos_feats
    }

    return g, feats, (net_map, pmos_map, nmos_map)


# ==========================================
#   2. HGAT 模型定义
# ==========================================

class HGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        # 定义异构卷积
        # 注意：如果某个图没有任何 PMOS，HeteroGraphConv 会自动处理空边的情况，但前提是输入特征维度正确
        self.conv = dglnn.HeteroGraphConv({
            'to_p': dglnn.GATConv(in_dim, out_dim // n_heads, num_heads=n_heads, allow_zero_in_degree=True),
            'to_n': dglnn.GATConv(in_dim, out_dim // n_heads, num_heads=n_heads, allow_zero_in_degree=True),
            'to_nm': dglnn.GATConv(in_dim, out_dim // n_heads, num_heads=n_heads, allow_zero_in_degree=True)
        }, aggregate='sum')

    def forward(self, g, h):
        h_out = self.conv(g, h)
        # Flatten heads
        return {k: v.flatten(1) for k, v in h_out.items()}


class HGATDesignEncoder(nn.Module):
    def __init__(self, in_dim_map, hid=64, out=64, n_layers=2, n_heads=4):
        super().__init__()
        self.hid = hid

        # Input Projections
        self.input_projs = nn.ModuleDict()
        for ntype, dim in in_dim_map.items():
            self.input_projs[ntype] = nn.Sequential(
                nn.Linear(dim, hid),
                nn.ReLU()
            )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HGATLayer(hid, hid, n_heads=n_heads))

        self.out_proj = nn.Linear(hid, out)

    def forward(self, g, feats):
        # 1. Feature Alignment
        h = {}
        for ntype, feat in feats.items():
            # 大小写兼容处理 (外部传入 NET, 内部用 net)
            key_upper = ntype.upper()
            if key_upper in self.input_projs:
                h[ntype] = self.input_projs[key_upper](feat)
            else:
                # 如果没有匹配的投影层（比如 data 里有 node_type 但 map 没定义），这理论上不该发生
                pass

        # 2. GNN Layers
        for layer in self.layers:
            h = layer(g, h)
            h = {k: F.elu(v) for k, v in h.items()}

        # 3. Readout (Pooling)
        with g.local_scope():
            g.ndata['h'] = h
            readouts = []
            for ntype in g.ntypes:
                if g.num_nodes(ntype) > 0:
                    # 使用 get 避免 key error
                    if ntype in h:
                        readouts.append(dgl.mean_nodes(g, 'h', ntype=ntype))

            if len(readouts) > 0:
                hg = torch.stack(readouts).mean(dim=0)
            else:
                # 极端情况：空图
                hg = torch.zeros(1, self.hid).to(next(self.parameters()).device)

            out = self.out_proj(hg)
            return out
