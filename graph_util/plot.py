import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")


def dglGraph_to_adj_list(g):
    adj_list = {}
    for node in range(g.number_of_nodes()):
        # For undirected graph. successors and
        # predecessors are equivalent.
        adj_list[node] = g.successors(node).tolist()
    return adj_list


class GraphPlot:
    def __init__(self, dir):
        super(GraphPlot, self).__init__()
        self.dir = dir

    def plot(self, adj_list, name):
        # to space out nodes
        pos = nx.spring_layout(
            nx.from_dict_of_lists(adj_list),
            k=0.8 * 1 / np.sqrt(len(nx.from_dict_of_lists(adj_list))),
            iterations=20,
        )
        nx.draw(
            nx.from_dict_of_lists(adj_list), node_size=500, with_labels=True, pos=pos
        )
        # plt.show()
        plt.savefig(self.dir + "/{:d}".format(name))
        plt.close()

    def plot_batch(self, adj_lists_to_plot):
        plot_times = 0

        if len(adj_lists_to_plot) >= 4:
            plot_times += 1
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
            axes = {0: ax0, 1: ax1, 2: ax2, 3: ax3}
            for i in range(1):
                nx.draw(
                    nx.from_dict_of_lists(adj_lists_to_plot[i]),
                    with_labels=True,
                    ax=axes[i],
                )
            # plt.show()
            plt.savefig(self.dir + "/{:d}".format(plot_times))
            # plt.close()
            adj_lists_to_plot = []

    def rollout_and_examine(self, model, num_samples):
        assert not model.training, "You need to call model.eval()."

        plot_times = 0
        adj_lists_to_plot = []

        for i in range(num_samples):
            sampled_graph = model()
            if isinstance(sampled_graph, list):
                # When the model is a batched implementation, a list of
                # DGLGraph objects is returned. Note that with model(),
                # we generate a single graph as with the non-batched
                # implementation. We actually support batched generation
                # during the inference so feel free to modify the code.
                sampled_graph = sampled_graph[0]

            print(sampled_graph)
            sampled_adj_list = dglGraph_to_adj_list(sampled_graph)
            adj_lists_to_plot.append(sampled_adj_list)
            print(sampled_adj_list)
            graph_size = sampled_graph.number_of_nodes()

            if len(adj_lists_to_plot) >= 4:
                plot_times += 1
                fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
                axes = {0: ax0, 1: ax1, 2: ax2, 3: ax3}
                for i in range(1):
                    nx.draw(
                        nx.from_dict_of_lists(adj_lists_to_plot[i]),
                        with_labels=True,
                        ax=axes[i],
                    )
                # plt.show()
                plt.savefig(self.dir + "/{:d}".format(plot_times))
                # plt.close()

                adj_lists_to_plot = []
