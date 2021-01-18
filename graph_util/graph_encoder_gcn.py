import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import TAGConv
from .graph_aggregate import GraphAggregate


class TAGCN(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, num_layers, activation, dropout, aggregation="none"
    ):
        super(TAGCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(TAGConv(hidden_dim, hidden_dim, activation=activation))
        # output layer
        # self.layers.append(TAGConv(hidden_dim, n_classes)) #activation=None
        self.dropout = nn.Dropout(p=dropout)

        self.aggregation = aggregation
        if aggregation == "weighted":
            self.G_aggregator = GraphAggregate(hidden_dim)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        if self.aggregation == "weighted":
            wh = self.G_aggregator(h)
            return wh
        else:
            return h


class GCN(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, num_layers, activation, dropout, aggregation="none"
    ):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        # output layer
        # self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

        self.aggregation = aggregation
        if aggregation == "weighted":
            self.G_aggregator = GraphAggregate(hidden_dim, "weighted")
        if aggregation == "attention":
            self.G_aggregator = GraphAggregate(hidden_dim, "attention")

    def forward(self, dgl_g, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(dgl_g, h)

        wh = self.G_aggregator(graph, h)
        return wh, h

    # else:
    #    return h
