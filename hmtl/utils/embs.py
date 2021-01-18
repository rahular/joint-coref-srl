import json
import torch

from tqdm import tqdm
from hmtl.utils import Graph, init_span_embedder, get_span_encoding
from hmtl.utils import K, get_config


def get_reps(graphs, desc):
    all_reps = []
    t = tqdm(
        graphs.items(),
        unit=" graphs",
        desc="Collecting {} graph representations".format(desc),
    )
    config = get_config()
    for dockey, graph in t:
        t.set_postfix({"Doc Index": dockey})
        if len(graph["nodes"]) < config["max_num_nodes"]:
            reps = get_span_encoding(
                dockey,
                zero_span_rep=config["span_rep_size"] if config.get("debug") else None,
            )
            all_spans = reps["all_spans"].cpu().detach().numpy()
            attended_span_embeddings = reps.get("attended_span_embeddings")
            endpoint_span_embeddings = reps.get("endpoint_span_embeddings")
            span_embeddings = (
                torch.cat([endpoint_span_embeddings, attended_span_embeddings], dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
            all_reps.append(
                (dockey, graph, all_spans, span_embeddings, reps["original_text"])
            )
    return all_reps


def get_roberta_reps(graphs, desc):
    all_reps = []
    t = tqdm(
        graphs.items(),
        unit=" graphs",
        desc="Collecting {} graph representations".format(desc),
    )
    config = get_config()
    for dockey, graph in t:
        t.set_postfix({"Doc Index": dockey})
        if len(graph["nodes"]) < config["max_num_nodes"]:
            reps = get_span_encoding(
                dockey,
                zero_span_rep=config["span_rep_size"] if config.get("debug") else None,
            )
            all_spans = reps["all_spans"].cpu().detach().numpy()
            roberta_embeddings = reps.get("roberta_embeddings")
            span_embeddings = roberta_embeddings.cpu().detach().numpy()
            all_reps.append(
                (dockey, graph, all_spans, span_embeddings, reps["original_text"])
            )
    return all_reps
