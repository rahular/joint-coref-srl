import torch
from torch import nn
import dgl

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class GraphAggregate(nn.Module):
    def __init__(self, node_hidden_size, aggregation_type):
        super(GraphAggregate, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = node_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(nn.Linear(node_hidden_size, 1), nn.Sigmoid())
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)

        self.type = aggregation_type

        # additive attention weights
        self.W_verbs = nn.Linear(node_hidden_size, node_hidden_size)
        self.W_nodes = nn.Linear(node_hidden_size, node_hidden_size)
        self.W_out = nn.Linear(node_hidden_size, 1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, graph, hvs):
        if self.type == "weighted":
            hvs = hvs
            whvs = self.node_gating(hvs) * self.node_to_graph(
                hvs
            )  # .sum(0, keepdim=True)
            return whvs
        elif self.type == "attention":
            nodes = graph["nodes"]
            verb_indices = [node["node_id"] for node in nodes if node["type"] == "V"]
            verb_reps = hvs[verb_indices]
            # print(verb_reps.shape, 'verb_reps.shape')
            verb_reps_aggregated = verb_reps.mean(dim=0)
            # print(verb_reps_aggregated.shape, " verb_reps_aggregated.shape")

            transformed_verb_reps = self.W_verbs(verb_reps_aggregated)
            # print(transformed_verb_reps.shape, "transformed_verb_reps")
            transformed_node_reps = self.W_nodes(hvs)
            # print(transformed_node_reps.shape, " transformed_node_reps")
            node_weights = self.softmax(
                self.W_out(
                    self.activation(transformed_verb_reps + transformed_node_reps)
                )
            )
            # print(node_weights.shape, "node_weights.shape")

            weighted_nodes = node_weights * hvs
            # print(weighted_nodes.shape, " weighted_nodes.shape")

            graph_rep = torch.sum(weighted_nodes, dim=0)
            # print(graph_rep.shape, "graph_rep")
            # print('\n')
            return graph_rep
