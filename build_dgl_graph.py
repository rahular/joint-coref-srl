import argparse
import itertools
import json
import os
import sys
import random
import logging
from collections import defaultdict
import collections

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate


from tqdm import tqdm

from graph_util.graph_encoder_gat import GAT
from graph_util.graph_encoder_gcn import GCN, TAGCN
from hmtl.training.loss import HingeLoss
from hmtl.utils import Graph, init_span_embedder, get_span_encoding, get_reps
from hmtl.utils import get_config, set_config, K

import dgl
from dgl import DGLError

import re
from torch._six import container_abcs, string_classes, int_classes

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


###### util ######
def set_device(d):
    # explicitly change `device`
    # can be used to setting device from other programs
    global device
    device = d


def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def array_in(arr, list_of_arr):
    for elem in list_of_arr:
        if (arr == elem).all():
            return True
    return False


def get_maximum_included_span(span, all_spans):
    start = span[0]
    for increment in range(1, 10):
        end = span[1] - increment
        new_span = [start, end]
        if array_in(np.array(new_span), all_spans):
            return new_span


def save_model(clf, G_encoder, clfNodes=None):
    config = get_config()
    torch.save(
        {
            "clf_state_dict": clf.state_dict(),
            "clfNodes": clfNodes.state_dict(),
            "G_encoder_state_dict": G_encoder.state_dict(),
        },
        os.path.join(
            config["reward_serialization_dir"],
            "{}_{}.th".format(args.model_name, args.encoder_type),
        ),
    )


def train_classifier(
    clf,
    clfNodes,
    G_encoder,
    train_data,
    val_data,
    neg_class,
    criterion=nn.L1Loss(),
    batch_size=32,
    epochs=100,
    patience=sys.maxsize,
    init_num_perturb_actions=1,
    clf_dataset_type="",
):

    optimizer = Adam(itertools.chain(G_encoder.parameters(), clf.parameters()))
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    best_f1, best_acc, best_epoch, patience_ctr = 0, 0, 0, -1

    def get_train_loader(epoch):
        if args.perturbation_type == "epoch":
            return DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            if clf_dataset_type == "combined":
                train_dataset = CombinedDataset(
                    G_encoder, train_data, neg_class, epoch, init_num_perturb_actions
                )
                return DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=my_collate,
                )
            else:
                train_dataset = GraphDataset(
                    G_encoder, train_data, neg_class, epoch, init_num_perturb_actions
                )
                return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def get_val_loaders():
        if args.perturbation_type == "epoch":
            return {
                "val_loader_epoch": DataLoader(
                    val_data, batch_size=batch_size, shuffle=False
                )
            }
        else:
            if clf_dataset_type == "combined":
                return {
                    n: DataLoader(
                        CombinedDataset(
                            G_encoder,
                            val_data,
                            neg_class,
                            decay_factor,
                            init_num_perturb_actions,
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=my_collate,
                    )
                    for n, decay_factor in (
                        ("val_loader_03", 0),
                        ("val_loader_02", 50),
                        ("val_loader_01", 100),
                    )
                }
            else:
                return {
                    n: DataLoader(
                        GraphDataset(
                            G_encoder,
                            val_data,
                            neg_class,
                            decay_factor,
                            init_num_perturb_actions,
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                    )
                    for n, decay_factor in (
                        ("val_loader_03", 0),
                        ("val_loader_02", 50),
                        ("val_loader_01", 100),
                    )
                }

    val_loaders = get_val_loaders()
    train_loader = get_train_loader(50)
    for epoch in range(epochs):
        # train_loader = get_train_loader(epoch)
        clf.train()
        clfNodes.train()
        G_encoder.train()
        running_loss = 0.0
        for data_item in train_loader:
            if clf_dataset_type == "combined":
                graph_x, graph_y, nodes_x, nodes_y = data_item
            else:
                graph_x, graph_y = data_item

            optimizer.zero_grad()
            graph_x = graph_x.to(device)
            graph_y = graph_y.to(device)
            logits_graph = clf(graph_x).squeeze()
            loss = criterion(logits_graph, graph_y.float())

            if clf_dataset_type == "combined":
                nodes_x = nodes_x.to(device)
                nodes_y = nodes_y.to(device)
                logits_nodes = clfNodes(nodes_x).squeeze()
                loss_nodes = criterion(logits_nodes, nodes_y.float())
                loss = loss + loss_nodes.mean()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        graph_avg_val_f1, graph_avg_val_acc = 0, 0
        nodes_avg_val_f1, nodes_avg_val_acc = 0, 0

        for name, val_loader in val_loaders.items():
            results = validate_classifier(
                clf,
                clfNodes,
                G_encoder,
                val_loader,
                neg_class=neg_class,
                clf_dataset_type=clf_dataset_type,
            )
            graph_acc = results.get("graph_acc", -1)
            graph_f1 = results.get("graph_f1", -1)
            nodes_acc = results.get("nodes_acc", -1)
            nodes_f1 = results.get("nodes_f1", -1)
            if clf_dataset_type == "combined":
                logging.info(
                    "After epoch {}: Validating {}... graph_val_acc: {}, graph_val_f1: {}, nodes_val_acc: {}, nodes_val_f1: {}".format(
                        epoch, name, graph_acc, graph_f1, nodes_acc, nodes_f1
                    )
                )
                nodes_avg_val_acc += nodes_acc
                nodes_avg_val_f1 += nodes_f1
            else:
                logging.info(
                    "After epoch {}: Validating {}... graph_val_acc: {}, graph_val_f1: {}".format(
                        epoch, name, graph_acc, graph_f1
                    )
                )
            graph_avg_val_acc += graph_acc
            graph_avg_val_f1 += graph_f1

        avg_val_acc = graph_avg_val_acc / len(val_loaders)
        avg_val_f1 = graph_avg_val_f1 / len(val_loaders)
        if clf_dataset_type == "combined":
            nodes_avg_val_acc = nodes_avg_val_acc / len(val_loaders)
            nodes_avg_val_f1 = nodes_avg_val_f1 / len(val_loaders)
            avg_val_acc = (avg_val_acc + nodes_avg_val_acc) / 2
            avg_val_f1 = (avg_val_f1 + nodes_avg_val_f1) / 2

        scheduler.step()
        if best_f1 < avg_val_f1:
            save_model(clf, G_encoder, clfNodes)
            best_f1 = avg_val_f1
            best_acc = avg_val_acc
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            logging.info("Ran out of patience. Stopping training...")
            break
    if clf_dataset_type == "combined":
        logging.info(
            "Best epoch: {}, (global/node) Best accuracy: {}/{}, Best f1: {}/{}".format(
                best_epoch, avg_val_acc, nodes_avg_val_acc, avg_val_f1, nodes_avg_val_f1
            )
        )
    else:
        logging.info(
            "Best epoch: {}, Best accuracy: {}, Best f1: {}".format(
                best_epoch, best_acc, best_f1
            )
        )


def validate_classifier(
    clf, clfNodes, G_encoder, loader, neg_class=0, batch_size=32, clf_dataset_type=""
):
    clf.eval()
    clfNodes.eval()
    G_encoder.eval()

    acc_total, acc, f1_total, f1, = 0, 0, 0, 0
    acc_total_nodes, acc_nodes, f1_total_nodes, f1_nodes, = 0, 0, 0, 0

    for data_item in loader:
        if clf_dataset_type == "combined":
            graph_x, graph_y, nodes_x, nodes_y = data_item
        else:
            graph_x, graph_y = data_item

        graph_x = graph_x.to(device)
        graph_y = graph_y.to(device)
        graph_logits = clf(graph_x).squeeze()

        if neg_class == -1:
            graph_logits[graph_logits >= 0] = 1
            graph_logits[graph_logits < 0] = -1
        elif neg_class == 0:
            graph_logits[graph_logits >= 0.5] = 1
            graph_logits[graph_logits < 0.5] = 0
        else:
            raise ValueError("Invalid `neg_class`.")
        acc += (
            torch.eq(graph_logits.detach().cpu(), graph_y.detach().float().cpu())
            .sum()
            .item()
        )
        acc_total += graph_y.size(0)
        f1 += f1_score(
            graph_y.detach().float().cpu().numpy(),
            graph_logits.detach().cpu().numpy(),
            average="binary",
        )
        f1_total += 1

        if clf_dataset_type == "combined":
            nodes_x = nodes_x.to(device)
            nodes_y = nodes_y.to(device)
            nodes_logits = clf(nodes_x).squeeze()

            if neg_class == -1:
                nodes_logits[nodes_logits >= 0] = 1
                nodes_logits[nodes_logits < 0] = -1
            elif neg_class == 0:
                nodes_logits[nodes_logits >= 0.5] = 1
                nodes_logits[nodes_logits < 0.5] = 0
            else:
                raise ValueError("Invalid `neg_class`.")
            acc_nodes += (
                torch.eq(nodes_logits.detach().cpu(), nodes_y.detach().float().cpu())
                .sum()
                .item()
            )
            acc_total_nodes += nodes_y.size(0)
            f1_nodes += f1_score(
                nodes_y.detach().float().cpu().numpy(),
                nodes_logits.detach().cpu().numpy(),
                average="binary",
            )
            f1_total_nodes += 1

    results = {"graph_acc": acc / acc_total, "graph_f1": f1 / f1_total}
    if clf_dataset_type == "combined":
        results["nodes_acc"] = acc_nodes / acc_total_nodes
        results["nodes_f1"] = f1_nodes / f1_total_nodes
    return results


def get_graph_reps(G_encoder, graph, all_spans, span_embeddings):
    config = get_config()

    G = GraphNN()
    G = G.to(device)
    nodes = graph["nodes"]
    edges = graph["edges"]
    # verb_indices = [node['node_id'] for node in nodes if node['type'] == 'V']
    root_idx = G.build_graph(nodes, edges, all_spans, span_embeddings)
    feats = G.g.ndata["hv"]
    graph_rep, node_reps = G_encoder(G.g, graph, feats)
    return graph_rep, node_reps

    """
    verb_reps = nodes_matrix_rep[verb_indices]
    verb_reps_aggregated = verb_reps.sum(dim=0)
    if encoding_type == 'root':
        return nodes_matrix_rep[root_idx]
    elif encoding_type == 'verb_nodes':
        return verb_reps_aggregated
    elif encoding_type == 'root_and_verbs':
        return torch.cat((nodes_matrix_rep[root_idx], verb_reps_aggregated))
    """


def perturb(
    dockey,
    graph_json,
    all_spans,
    span_embeddings,
    original_text,
    decay_factor,
    init_num_perturb_actions,
):
    config = get_config()

    ratio = (
        K.perturbation_ratio
        if args.perturbation_type == "all"
        else (np.array(K.perturbation_types) == args.perturbation_type).astype(float)
    )  # 1 for chosen, 0 for others
    graph, _ = Graph.build_graph(
        dockey,
        graph_json,
        max_span_width=config["max_span_width"],
        all_spans=all_spans,
        span_embeddings=span_embeddings,
        original_text=original_text,
    )
    # i.i.e perturbation probability for each node: on average we want k per sentence, decayed as training progresses
    num_sent = graph.metadata.get("num_sent", len(graph.nodes))
    graph.perturb(
        perturbation_probabilities=ratio / ratio.sum(),
        no_perturb_actions=int(
            np.ceil(init_num_perturb_actions * pow(0.99, decay_factor) * num_sent)
        ),
    )

    return graph.to_json()


def init_args():
    parser = argparse.ArgumentParser(
        description="Build node representation matrices for documents"
    )
    parser.add_argument("config_path", help="Configuration file")
    parser.add_argument("encoder_type", help="Type of graph encoder")
    parser.add_argument("perturbation_type", help="Type of perturbation to perform")
    parser.add_argument("model_name", help="Name for the model to be saved")
    parser.add_argument(
        "train_path",
        nargs="?",
        default="./models/coref_srl_conll_bert/graphs/graphs_gold_combined_train.json",
        help="JSON file containing training graphs",
    )
    parser.add_argument(
        "dev_path",
        nargs="?",
        default="./models/coref_srl_conll_bert/graphs/graphs_gold_combined_dev.json",
        help="JSON file containing validation graphs",
    )
    parser.add_argument(
        "neg_train_path",
        nargs="?",
        default="./models/coref_srl_conll_bert/graphs/graphs_epoch_1_combined_train.json",
        help="JSON file containing training graphs",
    )
    parser.add_argument(
        "neg_dev_path",
        nargs="?",
        default="./models/coref_srl_conll_bert/graphs/graphs_epoch_1_combined_dev.json",
        help="JSON file containing validation graphs",
    )
    return parser.parse_args()


###### classes ######
class GraphNN(nn.Module):
    def __init__(self):
        super(GraphNN, self).__init__()
        config = get_config()
        self.g = dgl.DGLGraph()

        # embedding and hidden layer sizes
        self.text_rep_size = config["span_rep_size"]
        self.node_hidden_size = config["node_hidden_size"]
        self.node_type_embedding_size = config["node_type_embedding_size"]
        self.edge_type_embedding_size = config["edge_type_embedding_size"]

        # embeddings and projection layer
        self.node_type_embeddings = nn.Embedding(
            len(K.id2node_type), self.node_type_embedding_size
        )
        self.node_type_embeddings.requires_grad = True
        self.projection_layer_node_init = nn.Linear(
            self.node_type_embedding_size + self.text_rep_size, self.node_hidden_size
        )
        self.edge_type_embeddings = nn.Embedding(
            len(K.id2edge_type), self.edge_type_embedding_size
        )
        self.edge_type_embeddings.requires_grad = True

        # edge list so far
        self.edge_tracker = defaultdict(list)

    def add_node(self, node_type, text_encoding=None):
        # add node
        self.g.add_nodes(1)
        num_nodes = self.g.number_of_nodes()

        # type embedding
        node_type_tensor = torch.LongTensor([K.node_type2id[node_type]]).to(
            device=device
        )
        type_embedding = self.node_type_embeddings(node_type_tensor).to(device=device)

        if node_type != "ROOT":
            # convert encoding to tensor
            text_encoding = torch.Tensor(text_encoding).unsqueeze(0).to(device)

            init_vector = torch.cat([type_embedding, text_encoding], -1)
            # init_vector = type_embedding
            init_vector_transformed = self.projection_layer_node_init(init_vector)

            # initial node rep
            self.g.nodes[num_nodes - 1].data["hv"] = init_vector_transformed
        else:
            # initial node rep
            self.g.nodes[num_nodes - 1].data["hv"] = type_embedding
        return num_nodes - 1

    def add_edge(
        self,
        edge_type,
        source_id,
        destination_id,
        bidirectional=False,
        reverse_dir=True,
    ):

        # for corr edges, bidir makes sense
        if bidirectional or edge_type == "cor":
            edge_type_tensor = torch.LongTensor([K.edge_type2id[edge_type]]).to(
                device=device
            )
            edge_type_embedding = self.edge_type_embeddings(edge_type_tensor).to(
                device=device
            )
            # repeat along first dim because one edge per direction
            edge_type_embedding_bidir = edge_type_embedding.repeat(2, 1)
            self.g.add_edges([source_id, destination_id], [destination_id, source_id])
            # this if is for the strange cases when the cor egde is the same as an exisiting srl edge
            if (destination_id, source_id) in self.edge_tracker.keys() or (
                source_id,
                destination_id,
            ) in self.edge_tracker.keys():
                no_exisitng_edges = (
                    self.g.edges[
                        [source_id, destination_id], [destination_id, source_id]
                    ]
                    .data["he"]
                    .shape[0]
                )
                self.g.edges[
                    [source_id, destination_id], [destination_id, source_id]
                ].data["he"][no_exisitng_edges - 2 :] = edge_type_embedding_bidir
            else:
                try:
                    self.g.edges[
                        [source_id, destination_id], [destination_id, source_id]
                    ].data["he"] = edge_type_embedding_bidir
                except DGLError as e:
                    raise RuntimeError(
                        "Failed adding edge with source_id={}, destination_id={}".format(
                            source_id, destination_id
                        )
                    ) from e

            self.edge_tracker[(source_id, destination_id)].append(edge_type)
            self.edge_tracker[(destination_id, source_id)].append(edge_type)

        else:
            edge_type_tensor = torch.LongTensor([K.edge_type2id[edge_type]]).to(
                device=device
            )
            edge_type_embedding = self.edge_type_embeddings(edge_type_tensor).to(
                device=device
            )

            # srl and root edges go from V -> and Root -> in original graph
            if reverse_dir:
                self.g.add_edge(destination_id, source_id)
                """
                # this is if for the strange cases when the cor egde is the same as an exisiting srl edge
                if (destination_id, source_id) in self.edge_tracker.keys():
                    # assume one dir is srl, the second is corr
                    edge_type_tensor_srl = torch.LongTensor([edge_type2id['srl']]).to(device=device)
                    edge_type_tensor_cor = torch.LongTensor([edge_type2id['cor']]).to(device=device)
                    edge_type_embedding_srl = self.edge_type_embeddings(edge_type_tensor_srl).to(device=device)
                    edge_type_embedding_cor = self.edge_type_embeddings(edge_type_tensor_cor).to(device=device)
                    edge_type_embedding = torch.cat([edge_type_embedding_srl, edge_type_embedding_cor], 0)

                    self.g.edges[destination_id, source_id].data['he'] = edge_type_embedding
                else:
                """
                self.g.edges[destination_id, source_id].data["he"] = edge_type_embedding

            else:
                self.g.add_edge(source_id, destination_id)
                """
                 #this is if for the strange cases when the cor egde is the same as an exisiting srl edge
                if  (source_id, destination_id) in self.edge_tracker.keys():
                    #assume one dir is srl, the second is corr
                    edge_type_tensor_srl = torch.LongTensor([edge_type2id['srl']]).to(device=device)
                    edge_type_tensor_cor = torch.LongTensor([edge_type2id['cor']]).to(device=device)
                    edge_type_embedding_srl = self.edge_type_embeddings(edge_type_tensor_srl).to(device=device)
                    edge_type_embedding_cor = self.edge_type_embeddings(edge_type_tensor_cor).to(device=device)
                    edge_type_embedding = torch.cat([edge_type_embedding_srl, edge_type_embedding_cor], 0)
                    self.g.edges[source_id, destination_id].data['he'] = edge_type_embedding
                else:
                """
                self.g.edges[source_id, destination_id].data["he"] = edge_type_embedding

            if reverse_dir:
                self.edge_tracker[(destination_id, source_id)].append(edge_type)
            else:
                self.edge_tracker[(source_id, destination_id)].append(edge_type)

    def get_encoded_text(self, doc_indices, all_spans, span_embeddings):
        # clamp span len to 10
        if doc_indices[1] - doc_indices[0] > 9:
            new_doc_indices = []
            new_doc_indices.append(doc_indices[0])
            new_doc_indices.append(doc_indices[0] + 9)
        else:
            new_doc_indices = doc_indices

        if not array_in(np.array(new_doc_indices), all_spans[0]):
            new_doc_indices = get_maximum_included_span(new_doc_indices, all_spans[0])

        index_in_all_spans = [
            index
            for index, item in enumerate(all_spans[0])
            if list(item) == list(new_doc_indices)
        ][0]

        if span_embeddings.ndim == 2:
            span_embeddings = np.expand_dims(span_embeddings, axis=0)

        text_encoded = span_embeddings[0][index_in_all_spans]
        return text_encoded

    def build_graph(self, nodes_list, edges_list, all_spans, span_embeddings):
        for node in nodes_list:
            if node["type"] != "ROOT":
                text_encoded = self.get_encoded_text(
                    node["doc_indices"], all_spans, span_embeddings
                )
                self.add_node(node["type"], text_encoded)
            else:
                root_idx = self.add_node(node["type"])

        for edge in edges_list:
            source_id = edge["nids"][0]
            destination_id = edge["nids"][1]
            self.add_edge(edge["type"], source_id, destination_id)

        return root_idx


class ClfHead(nn.Module):
    def __init__(self, input_dim=200):
        super(ClfHead, self).__init__()
        self.lin1 = nn.Linear(input_dim, int(input_dim / 2))
        self.lin2 = nn.Linear(int(input_dim / 2), 1)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x)))


class CombinedDataset(Dataset):
    def __init__(
        self, G_encoder, graph_tuples, neg_class, decay_factor, init_num_perturb_actions
    ):
        self.G_encoder = G_encoder
        (
            self.dockeys,
            self.graphs,
            self.all_spans,
            self.span_embeddings,
            self.original_text,
        ) = list(zip(*graph_tuples))
        self.neg_class = neg_class
        self.decay_factor = decay_factor
        self.init_num_perturb_actions = init_num_perturb_actions

    def __getitem__(self, idx):
        real_idx = int(idx / 2)
        graph = self.graphs[real_idx]
        # add not perturbed as default
        for id, node in enumerate(graph["nodes"]):
            graph["nodes"][id]["perturbed"] = False
        all_spans = self.all_spans[real_idx][0]
        span_embeddings = self.span_embeddings[real_idx][0]
        original_text = self.original_text[real_idx]
        if idx % 2 == 0:
            graph_y = 1
        else:
            graph = perturb(
                self.dockeys[real_idx],
                graph,
                all_spans,
                span_embeddings,
                original_text,
                self.decay_factor,
                self.init_num_perturb_actions,
            )
            graph_y = self.neg_class
        graph_x, nodes_X = get_graph_reps(
            self.G_encoder, graph, self.all_spans[real_idx], span_embeddings
        )
        # get per node perturbation labels
        nodes = graph["nodes"]
        if self.neg_class == -1:
            nodes_Y = torch.Tensor([-1 if n["perturbed"] else 1 for n in nodes])
        elif self.neg_class == 0:
            nodes_Y = torch.Tensor([0 if n["perturbed"] else 1 for n in nodes])
        nodes_X = nodes_X
        # expand first dim of G
        graph_x = graph_x.unsqueeze(0)
        graph_y = graph_y
        return graph_x, graph_y, nodes_X, nodes_Y

    def __len__(self):
        return len(self.graphs) * 2


class GraphDataset(Dataset):
    def __init__(
        self, G_encoder, graph_tuples, neg_class, decay_factor, init_num_perturb_actions
    ):
        self.G_encoder = G_encoder

        (
            self.dockeys,
            self.graphs,
            self.all_spans,
            self.span_embeddings,
            self.original_text,
        ) = list(zip(*graph_tuples))
        self.neg_class = neg_class
        self.decay_factor = decay_factor
        self.init_num_perturb_actions = init_num_perturb_actions

    def __getitem__(self, idx):
        real_idx = int(idx / 2)
        graph = self.graphs[real_idx]
        # add not perturbed as default
        for id, node in enumerate(graph["nodes"]):
            graph["nodes"][id]["perturbed"] = False
        all_spans = self.all_spans[real_idx][0]
        span_embeddings = self.span_embeddings[real_idx][0]
        original_text = self.original_text[real_idx]
        if idx % 2 == 0:
            graph_y = 1
        else:
            graph = perturb(
                self.dockeys[real_idx],
                graph,
                all_spans,
                span_embeddings,
                original_text,
                self.decay_factor,
                self.init_num_perturb_actions,
            )
            graph_y = self.neg_class

        graph_x, _ = get_graph_reps(
            self.G_encoder, graph, self.all_spans[real_idx], span_embeddings
        )
        graph_y = graph_y

        return graph_x, graph_y

    def __len__(self):
        return len(self.graphs) * 2


class NodesDataset(Dataset):
    def __init__(
        self, G_encoder, graph_tuples, neg_class, decay_factor, init_num_perturb_actions
    ):
        self.G_encoder = G_encoder
        (
            self.dockeys,
            self.graphs,
            self.all_spans,
            self.span_embeddings,
            self.original_text,
        ) = list(zip(*graph_tuples))
        self.neg_class = neg_class
        self.decay_factor = decay_factor
        self.init_num_perturb_actions = init_num_perturb_actions

    def __getitem__(self, idx):
        real_idx = int(idx / 2)
        graph = self.graphs[real_idx]
        # add not perturbed as default
        for id, node in enumerate(graph["nodes"]):
            graph["nodes"][id]["perturbed"] = False

        all_spans = self.all_spans[real_idx][0]
        span_embeddings = self.span_embeddings[real_idx][0]
        original_text = self.original_text[real_idx]
        if idx % 2 == 0:
            graph_y = 1
        else:
            graph = perturb(
                self.dockeys[real_idx],
                graph,
                all_spans,
                span_embeddings,
                original_text,
                self.decay_factor,
                self.init_num_perturb_actions,
            )
            graph_y = self.neg_class
        graph_x, nodes_X = get_graph_reps(
            self.G_encoder, graph, self.all_spans[real_idx], span_embeddings
        )

        # get per node perturbation labels
        nodes = graph["nodes"]

        if self.neg_class == -1:
            nodes_Y = torch.Tensor([-1 if n["perturbed"] else 1 for n in nodes])
        elif self.neg_class == 0:
            nodes_Y = torch.Tensor([0 if n["perturbed"] else 1 for n in nodes])
        nodes_X = nodes_X
        return nodes_X, nodes_Y

    def __len__(self):
        return len(self.graphs) * 2


class VanillaDataset(Dataset):
    def __init__(self, G_encoder, x, y):
        (
            self.dockeys,
            self.graphs,
            self.all_spans,
            self.span_embeddings,
            self.original_text,
        ) = list(zip(*x))
        self.G_encoder = G_encoder
        self.normalizer = Normalizer()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        graph_x, _ = get_graph_reps(
            self.G_encoder,
            self.graphs[idx],
            self.all_spans[idx],
            self.span_embeddings[idx][0],
        )
        return (
            np.squeeze(self.normalizer.transform(graph_x.view(-1, 1).detach().numpy())),
            self.y[idx],
        )

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    args = init_args()
    criterion = nn.MSELoss()
    neg_class = (
        -1
        if isinstance(criterion, HingeLoss) or criterion == nn.HingeEmbeddingLoss
        else 0
    )
    config = set_config(args.config_path)

    if args.encoder_type == "GAT":
        G_encoder = GAT(
            config["node_hidden_size"],
            config["node_hidden_size"],
            config["num_graph_heads"],
            merge=config["graph_head_merge_type"],
            aggregation=config["graph_aggregation"],
        )

    elif args.encoder_type == "GCN":
        G_encoder = GCN(
            config["node_hidden_size"],
            config["node_hidden_size"],
            config["num_gencoder_layers"],
            None,
            config["gencoder_dropout"],
            aggregation=config["graph_aggregation"],
        )

    elif args.encoder_type == "TAGCN":
        G_encoder = TAGCN(
            config["node_hidden_size"],
            config["node_hidden_size"],
            config["num_gencoder_layers"],
            None,
            config["gencoder_dropout"],
            aggregation=config["graph_aggregation"],
        )
    else:
        raise ValueError("Invalid graph encoder type")

    logging.info("Using Encoder: {}".format(args.encoder_type))
    G_encoder = G_encoder.to(device)

    if config["graph_aggregation"] == "weighted":
        logging.info("Using weighted sum graph aggregator")

    init_span_embedder(
        config["pretrained_serialization_dir"], config["combined_train_json_path"]
    )
    with open(args.train_path, encoding="utf-8") as f:
        train_graphs = json.load(f)
    train_reps = get_reps(train_graphs, "training")

    init_span_embedder(
        config["pretrained_serialization_dir"], config["combined_dev_json_path"]
    )
    with open(args.dev_path, encoding="utf-8") as f:
        val_graphs = json.load(f)
    val_reps = get_reps(val_graphs, "validation")

    if args.perturbation_type == "epoch":
        init_span_embedder(
            config["pretrained_serialization_dir"],
            config["neg_combined_train_json_path"],
        )
        with open(args.neg_train_path, encoding="utf-8") as f:
            neg_train_graphs = json.load(f)
        neg_train_reps = get_reps(neg_train_graphs, "training")

        init_span_embedder(
            config["pretrained_serialization_dir"], config["neg_combined_dev_json_path"]
        )
        with open(args.neg_dev_path, encoding="utf-8") as f:
            neg_val_graphs = json.load(f)
        neg_val_reps = get_reps(neg_val_graphs, "validation")

        train_reps = VanillaDataset(
            G_encoder,
            train_reps + neg_train_reps,
            [1] * len(train_reps) + [neg_class] * len(neg_train_reps),
        )
        val_reps = VanillaDataset(
            G_encoder,
            val_reps + neg_val_reps,
            [1] * len(val_reps) + [neg_class] * len(neg_val_reps),
        )

    logging.info("Training classifier...")
    input_dim = config["node_hidden_size"]
    if config["graph_head_merge_type"] == "cat":
        input_dim *= 3

    clf = ClfHead(input_dim=input_dim)
    clfNodes = ClfHead(input_dim=input_dim)

    clf = clf.to(device)
    clfNodes = clfNodes.to(device)

    train_classifier(
        clf,
        clfNodes,
        G_encoder,
        train_reps,
        val_reps,
        neg_class=neg_class,
        criterion=criterion,
        patience=config["reward_training_patience"],
        init_num_perturb_actions=config["init_num_perturb_actions"],
        clf_dataset_type=config["clf_dataset_type"],
    )
