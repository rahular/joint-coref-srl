"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import torch.nn.functional as F

from .graph_aggregate import GraphAggregate


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata["z"] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        elif self.merge == "max":
            # max pool
            max_pooled, _ = torch.max(torch.stack(head_outs), dim=0)
            return max_pooled
        elif self.merge == "mean":
            # merge using average
            return torch.mean(torch.stack(head_outs))
        else:
            raise ValueError("Invalid `merge` value.")


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, merge="max", aggregation="none"):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, merge)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        if merge == "cat":
            self.layer2 = MultiHeadGATLayer(
                hidden_dim * num_heads, hidden_dim, 1, merge
            )
        else:
            self.layer2 = MultiHeadGATLayer(hidden_dim, hidden_dim, 1, merge)

        self.aggregation = aggregation
        if aggregation == "weighted":
            self.G_aggregator = GraphAggregate(hidden_dim)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)

        if self.aggregation == "weighted":
            wh = self.G_aggregator(h)
            return wh
        else:
            return h
