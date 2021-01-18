"""
Requires torch-sparse, see instructions here:
https://github.com/rusty1s/pytorch_geometric/issues/1001#issuecomment-598708757

Get preprocessed files from:
/science/image/nlp-letre/e2e-allennlp/coref_srl_conll_bert
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
"""

import json
import copy
import logging
import pickle
import argparse
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax, RGCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from hmtl.utils import Graph, init_span_embedder, get_span_encoding, get_reps
from hmtl.utils import K, get_config, set_config

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_types = dict()

# seeds
torch.manual_seed(42)
np.random.seed(42)
if device.type != "cpu":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_device(d):
    # this is invoked from the fine-tuning code when reward models
    # need to be moved to the CPU
    global device
    device = d


def graph2pytg(graph):
    invalid_nodes = 0
    max = graph.max_span_width

    # TODO: need to deal with root node which has no spans or which have invalid spans
    # (like those which span across sentence boundaries)
    dummy_span = torch.tensor(np.zeros_like(graph.span_embeddings[0]))
    span_lengths = [span[1] - span[0] if span else 0 for span in graph.node_spans]
    corrected_indices = []
    for span, span_len in zip(graph.node_spans, span_lengths):
        if not span or span == (-1, -1):
            # control comes here only for the root node and hence
            # it is not counted as an invalid node
            corrected_indices.append((0, 0))
        else:
            if span_len < max:
                start, end = span[0], span[1]
            else:
                start, end = span[0], span[0] + max - 1
            if (start, end) not in graph.all_spans:
                logging.info("Invalid span: ({}, {})".format(start, end))
                invalid_nodes += 1
                corrected_indices.append((0, 0))
            else:
                corrected_indices.append((start, end))

    span_reps = [
        graph.span_embeddings_by_span[span] if span != (0, 0) else dummy_span
        for span in corrected_indices
    ]
    if invalid_nodes > 0:
        logging.info(
            "Found {}/{} invalid nodes.".format(invalid_nodes, len(graph.node_spans))
        )

    # get ys for validation task
    y = list()
    for _, node in graph.nodes.items():
        if node.type in node_types:
            node_type = node_types[node.type]
        else:
            node_type = node_types[node.type] = len(node_types)
        y.append(node_type)

    # add node types and spans to node init
    type_features = [type for type in y]
    span_reps = torch.tensor(np.stack(span_reps))
    type_features = torch.tensor(np.stack(type_features))

    # get edges and edge attributes
    edges, edge_types = graph.get_edges()
    edge_index = torch.tensor(edges).permute(1, 0)
    edge_attr = torch.tensor([1 if e == "cor" else 0 for e in edge_types])

    # get and set num_nodes
    num_nodes = len(graph.nodes)

    return Data(
        span_reps=span_reps.float(),
        type_features=type_features.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y),
        graph=graph,
        num_nodes=num_nodes,
    )


def perturb(graph, *args, **kwargs):
    perturbation_type = kwargs["perturbation_type"]
    decay_factor = kwargs["decay_factor"] * 3
    ratio = (
        K.perturbation_ratio
        if perturbation_type == "all"
        else (np.array(K.perturbation_types) == perturbation_type).astype(float)
    )  # 1 for chosen, 0 for others

    # i.i.e perturbation probability for each node: on average we want k per sentence,
    # decayed as training progresses
    num_sent = graph.metadata.get("num_sent", len(graph.nodes))
    graph.perturb(
        perturbation_probabilities=ratio / ratio.sum(),
        no_perturb_actions=int(np.ceil(pow(0.8, decay_factor) * num_sent)),
    )
    return graph


def read_data(pretrained_serialization_dir, set_name):
    config = get_config()
    combined_path = "graphs/gold_combined_{}.json".format(set_name)
    combined_graph_path = "graphs/graphs_gold_combined_{}.json".format(set_name)
    with open(
        os.path.join(pretrained_serialization_dir, combined_graph_path), "r"
    ) as f:
        graph_data = json.load(f)

    init_span_embedder(pretrained_serialization_dir, combined_path)
    reps = get_reps(graph_data, set_name)

    data_list = []
    for dockey, gdata, all_spans, span_embeddings, original_text in reps:
        graph, _ = Graph.build_graph(
            dockey,
            gdata,
            max_span_width=config["max_span_width"],
            all_spans=all_spans[0],
            span_embeddings=span_embeddings[0],
            original_text=original_text,
        )
        if len(graph.nodes) > 0:
            data_list.append(graph2pytg(graph))
        num_features = (
            span_embeddings[0][0].shape[0] + config["node_type_embedding_size"]
        )
    return DataLoader(data_list, batch_size=1), num_features


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, node_emb_size, encoder_type="GCN"):
        super(Encoder, self).__init__()
        self.enc_type = encoder_type
        if self.enc_type == "GCN":
            self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        elif self.enc_type == "RGCN":
            self.conv = RGCNConv(
                in_channels, hidden_channels, num_relations=3, num_bases=2
            )
        self.prelu = nn.PReLU(hidden_channels)
        self.node_type_embedding_size = node_emb_size
        self.node_type_embeddings = nn.Embedding(
            len(K.id2node_type), self.node_type_embedding_size
        )
        self.node_type_embeddings.requires_grad = True

    def forward(self, datapoints, *args, **kwargs):
        if type(datapoints) is DataLoader:
            datapoints_batch = [d for d in datapoints][0]
        else:
            datapoints_batch = datapoints

        span_reps = datapoints_batch.span_reps.to(device)
        node_type_features = datapoints_batch.type_features.to(device)
        edge_type_features = datapoints_batch.edge_attr.to(device)

        # embed node type
        node_type_embeddings = self.node_type_embeddings(node_type_features.long())
        xs = torch.cat([span_reps, node_type_embeddings], dim=-1)
        edge_indices = datapoints_batch.edge_index.to(device)
        if self.enc_type == "GCN":
            x = self.conv(xs.float(), edge_indices)
        elif self.enc_type == "RGCN":
            x = self.conv(xs.float(), edge_indices, edge_type_features)
        x = self.prelu(x)
        return x


def corruption(datapoints, *args, **kwargs):
    pytg_perturbed_graphs = []
    graphs = datapoints.graph
    if not isinstance(graphs, list):
        graphs = [graphs]
    for graph in graphs:
        perturbed_graph = perturb(graph, *args, **kwargs)
        pytg_perturbed_graph = graph2pytg(perturbed_graph)
        pytg_perturbed_graphs.append(pytg_perturbed_graph)
    return DataLoader(pytg_perturbed_graphs, batch_size=1)


def summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))


def train(loader, model, optimizer, perturbation_type=None, decay_factor=-1):
    model = model.train()
    model = model.float()
    cum_loss = 0.0
    for _, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(
            data, perturbation_type=perturbation_type, decay_factor=epoch
        )
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
    return cum_loss / len(loader.dataset)


def classify(
    train_z,
    train_y,
    test_z,
    test_y,
    f1_average="binary",
    solver="lbfgs",
    multi_class="auto",
    *args,
    **kwargs
):
    clf = LogisticRegression(
        solver=solver, multi_class=multi_class, *args, **kwargs
    ).fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
    preds = clf.predict(test_z.detach().cpu().numpy())
    acc = accuracy_score(test_y.detach().cpu().numpy(), preds)
    f1 = f1_score(test_y.detach().cpu().numpy(), preds, average=f1_average)
    return clf, acc, f1


def test_nodes(
    train_loader, dev_loader, model, perturbation_type=None, decay_factor=-1
):
    model = model.eval()
    train_z, train_y = [], []
    dev_z, dev_y = [], []
    for _, data in enumerate(train_loader):
        data = data.to(device)
        z, _, _ = model(data, perturbation_type=perturbation_type, decay_factor=epoch)
        train_z.append(z)
        train_y.append(data.y)
    for data in dev_loader:
        data = data.to(device)
        z, _, _ = model(data, perturbation_type=perturbation_type, decay_factor=epoch)
        dev_z.append(z)
        dev_y.append(data.y)
    train_z = torch.cat(train_z, dim=0)
    train_y = torch.cat(train_y, dim=0)
    dev_z = torch.cat(dev_z, dim=0)
    dev_y = torch.cat(dev_y, dim=0)

    return classify(train_z, train_y, dev_z, dev_y, f1_average="micro", max_iter=150)


def test_graphs(
    train_loader, dev_loader, model, perturbation_type=None, decay_factor=-1
):
    pos_class = torch.tensor([1])
    neg_class = torch.tensor([0])
    model = model.eval()
    train_z, train_y = [], []
    dev_z, dev_y = [], []
    for _, data in enumerate(train_loader):
        data = data.to(device)
        _, zn, sump = model(
            data, perturbation_type=perturbation_type, decay_factor=epoch
        )
        train_z.append(sump.unsqueeze(0))
        train_y.append(pos_class)
        train_z.append(summary(zn).unsqueeze(0))
        train_y.append(neg_class)
    for data in dev_loader:
        data = data.to(device)
        _, zn, sump = model(
            data, perturbation_type=perturbation_type, decay_factor=epoch
        )
        dev_z.append(sump.unsqueeze(0))
        dev_y.append(pos_class)
        dev_z.append(summary(zn).unsqueeze(0))
        dev_y.append(neg_class)
    train_z = torch.cat(train_z, dim=0)
    train_y = torch.cat(train_y, dim=0)
    dev_z = torch.cat(dev_z, dim=0)
    dev_y = torch.cat(dev_y, dim=0)

    return classify(train_z, train_y, dev_z, dev_y, max_iter=150)


def save(encoder_state, clf, optimizer_state, perturbation_type):
    config = get_config()
    save_path = os.path.join(
        config["reward_serialization_dir"],
        "best_dgi_with_decay_{}_model.th".format(perturbation_type),
    )
    blob = {
        "encoder_state": encoder_state,
        "classifier_state": pickle.dumps(clf),
        "training_state": optimizer_state,
    }
    torch.save(blob, save_path)


def init_args():
    parser = argparse.ArgumentParser(description="Train DGI reward models")
    parser.add_argument("config_path", help="Configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    config = set_config(args.config_path)
    train_loader, num_features = read_data(
        config["pretrained_serialization_dir"], "train"
    )
    dev_loader, _ = read_data(config["pretrained_serialization_dir"], "dev")

    for perturbation_type in K.perturbation_types:
        model = DeepGraphInfomax(
            hidden_channels=512,
            encoder=Encoder(num_features, 512, config["node_type_embedding_size"]),
            summary=summary,
            corruption=corruption,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        best_encoder_weights, best_clf, best_graph_acc, best_graph_f1 = None, None, 0, 0
        for epoch in range(3):
            loss = train(
                train_loader,
                model,
                optimizer,
                perturbation_type=perturbation_type,
                decay_factor=epoch,
            )
            # actual pertrubed vs. non-perturbed classification
            clf, graph_acc, graph_f1 = test_graphs(
                train_loader,
                dev_loader,
                model,
                perturbation_type=perturbation_type,
                decay_factor=epoch,
            )
            if best_graph_f1 < graph_f1:
                best_encoder_weights = copy.deepcopy(model.state_dict())
                best_clf = clf
                best_graph_acc = graph_acc
                best_graph_f1 = graph_f1
            logging.info(
                "Epoch: {:03d}, Loss: {:.4f}, F1: {:.4f}, Best F1: {:.4f}".format(
                    epoch + 1, loss, graph_f1, best_graph_f1
                )
            )
            save(
                best_encoder_weights, best_clf, optimizer.state_dict(), perturbation_type,
            )
        logger.info(
            "Finished {} for {} with F1 {}".format(
                perturbation_type, args.config_path, best_graph_f1
            )
        )
