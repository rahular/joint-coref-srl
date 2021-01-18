import os
import sys
import json
import math
import copy
import spacy
import torch
import time
import random
import logging
import traceback
import argparse
import pickle

from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from uuid import uuid4
from collections import defaultdict, deque
from torch.optim import AdamW
from torch_geometric.nn import GCNConv, DeepGraphInfomax, RGCNConv
from random import shuffle
from sklearn.linear_model import LogisticRegression

import build_dgl_graph as BDG
import dgi as DGI
from evaluate import evaluate
from graph_util import json2graph

from allennlp.nn import util
from allennlp.data.fields import MetadataField
from allennlp.data.tokenizers import Token
from allennlp.common.params import Params
from allennlp.training.optimizers import Optimizer

from hmtl.utils import Graph, init_model, init_pretrained_model
from hmtl.utils import K, set_config, get_config
from hmtl.dataset_readers import (
    CorefConllReader,
    CorefPrecoReader,
    CorefWikicorefReader,
    CorefWinobiasReader,
    SrlReader,
    SrlReader05,
    SrlWikibankReader,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CPU = torch.device("cpu")
GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# seeds
torch.manual_seed(42)
if GPU.type != "cpu":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_wiki_articles(path):
    logger.info("Reading wikipedia articles from disk")
    data = dict()
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            doc = nlp(line.strip())
            data[str(uuid4())] = doc
    return data


def get_coref_instances(articles, reader, model):
    ft_instances = dict()
    logger.info("Making coref instances from wikipedia articles")
    for doc_key, doc in tqdm(articles.items()):
        instance = reader.text_to_instance(
            [[tok.text for tok in sent] for sent in doc.sents]
        )
        instance.fields["metadata"].metadata["doc_key"] = doc_key
        instance.index_fields(model.vocab)
        ft_instances[doc_key] = instance
    return ft_instances


def get_srl_instances(articles, reader, model):
    ft_instances = defaultdict(list)
    logger.info("Making SRL instances from wikipedia articles")
    for doc_key, doc in tqdm(articles.items()):
        for idx, sent in enumerate(doc.sents):
            verbs = [
                (tok.text, tok.i - sent.start) for tok in sent if tok.pos_ == "VERB"
            ]
            for _, verb_loc in verbs:
                tokenized_sent = [Token(tok.text) for tok in sent]
                verb_indicator = [0] * len(tokenized_sent)
                verb_indicator[verb_loc] = 1
                instance = reader.text_to_instance(tokenized_sent, verb_indicator)
                instance.fields["metadata"].metadata["sent_key"] = "{}-{}".format(
                    doc_key, str(idx)
                )
                instance.index_fields(model.vocab)
                ft_instances[doc_key].append(instance)
    return ft_instances


def get_dev_instances(ft_task, ft_data, model):
    reader_classes = {
        "coref": {
            "conll12": CorefConllReader,
            "preco": CorefPrecoReader,
            "wikicoref": CorefWikicorefReader,
            "winobias": CorefWinobiasReader,
            "phrase_detectives_w": CorefConllReader,  # works just fine
            "phrase_detectives_g": CorefConllReader,  # works just fine
        },
        "srl": {
            "conll12": SrlReader,
            "conll05": SrlReader05,
            "wikibank": SrlWikibankReader,
            "ewt": SrlWikibankReader,  # works just fine
        },
    }
    params = Params.from_file(os.path.join(args.serialization_dir, "./config.json"))
    task_name = "task_{}".format(ft_task)
    reader_params = params.get(task_name)["data_params"].get(
        "validation_dataset_reader", None
    )
    if reader_params is None:
        reader_params = params.pop(task_name)["data_params"]["dataset_reader"]
    reader_params.pop("type")
    reader = reader_classes[ft_task][ft_data].from_params(reader_params)
    dev_instances = []
    for instance in reader._read(
        config["{}_{}_dev_data_path".format(ft_task, ft_data)]
    ):
        instance.index_fields(model.vocab)
        dev_instances.append(instance)
    return dev_instances


def get_logits(instances, iterator, model, task, sample):
    if task == "srl":
        instances = [i for sublist in instances for i in sublist]
    generator = iterator(instances=instances, num_epochs=1)
    outputs = []
    for batch in generator:
        # model is on gpu only if sample is true i.e. model == ft_model
        if sample and next(model.parameters()).is_cuda and type(GPU.index) == int:
            batch = util.move_to_device(batch, GPU.index)
        output = model.forward(
            tensor_batch=batch, task_name=task, for_training=True, sample=sample
        )
        output = model.decode(output, task_name=task)
        outputs.append(output)
    return outputs


def combine_outputs(coref_outputs, srl_outputs):
    output_dict = dict()
    for co in coref_outputs:
        doc_key = co["doc_key"][0]
        output_dict[doc_key] = {
            "document": co["document"][0],
            "clusters": co["clusters"],
            "all_spans": co["all_spans"],
            "endpoint_span_embeddings": co["endpoint_span_embeddings"],
            "attended_span_embeddings": co["attended_span_embeddings"],
            "coref_log_likelihood": co.get("log_likelihood", None),
        }
        output_dict[doc_key]["sentences"]: List(str) = []
        output_dict[doc_key]["srl_tags"]: List(dict) = []
        output_dict[doc_key]["srl_log_likelihood"] = 0

    sentences, srl_tags = {}, defaultdict(list)
    for so in srl_outputs:
        for words, verb, sent_key, tags in zip(
            so["words"], so["verb"], so["sent_key"], so["tags"]
        ):
            sentences[sent_key] = words
            srl_tags[sent_key].append({"verb": verb, "tags": tags})
        if "log_likelihood" in so:
            doc_key = "-".join(sent_key.split("-")[:-1])
            output_dict[doc_key]["srl_log_likelihood"] = so["log_likelihood"]

    for sent_key in sentences:
        sentence = sentences[sent_key]
        tags = srl_tags[sent_key]
        doc_key = "-".join(sent_key.split("-")[:-1])
        output_dict[doc_key]["sentences"].append(sentence)
        output_dict[doc_key]["srl_tags"].append(tags)
    return output_dict


def init_reward_models(
    serialization_dir, ft_task="coref", model_type=None, encoder_type="GCN"
):
    clfs = dict()
    if model_type == "sep":
        ptypes = (
            K.perturbation_types
            if config["use_all_type_rewards"]
            else [p for p in K.perturbation_types if ft_task in p]
        )
        for p in ptypes:
            try:
                clf, enc = init_reward_model(serialization_dir, p)
                clfs[p] = {"clf": clf, "enc": enc}
            except Exception as e:
                logger.error("Model for {} not found.".format(p))
                logger.error(e)
    elif model_type == "com":
        clf, enc = init_reward_model(serialization_dir, model_type)
        clfs[model_type] = {"clf": clf, "enc": enc}
    else:
        raise ValueError("Invalid `mode_type` option.")
    return clfs


def init_reward_model(serialization_dir, model_name):
    config = get_config()
    mpath = os.path.join(
        serialization_dir, "best_dgi_with_decay_{}_model.th".format(model_name)
    )
    logger.info("Loading reward models from {}".format(mpath))

    num_features = config["span_rep_size"] + config["node_type_embedding_size"]
    G_encoder = DeepGraphInfomax(
        hidden_channels=512,
        encoder=DGI.Encoder(num_features, 512, config["node_type_embedding_size"]),
        summary=DGI.summary,
        corruption=DGI.corruption,
    )
    state_dicts = torch.load(mpath, map_location=CPU)
    G_encoder.load_state_dict(state_dicts["encoder_state"])
    clf = pickle.loads(state_dicts["classifier_state"])
    return clf, G_encoder.eval().to(CPU)


def get_reward(clfs, graph):
    data = DGI.graph2pytg(graph)
    reward = 0.0
    for key in clfs:
        _, _, graph_rep = clfs[key]["enc"](
            data, perturbation_type="coref_add_ant", decay_factor=0
        )  # perturbation type does not matter. `coref_add_ant` is used because it is the fastest
        reward += clfs[key]["clf"].predict_proba(
            graph_rep.unsqueeze(0).detach().cpu().numpy()
        )[0][1]
    return reward / len(clfs)


def get_optimizer(model, ft_task):
    params = Params.from_file(os.path.join(args.serialization_dir, "./config.json"))
    optimizer_params = params["multi_task_trainer"]["optimizer"]
    parameters_to_train = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
    if ft_task == "coref":
        optimizer_params["lr"] = 3e-4
    optimizer = Optimizer.from_params(
        model_parameters=parameters_to_train, params=copy.deepcopy(optimizer_params)
    )
    return optimizer


def pg_loop(
    wiki_articles,
    artefacts,
    clfs,
    batch_size=5,
    epochs=1000,
    to_finetune="srl",
    finetune_dataset="conll12",
):
    coref_reader, coref_iterator, coref_model = (
        artefacts["readers"]["coref"],
        artefacts["iterators"]["coref"],
        artefacts["models"]["coref"],
    )
    srl_reader, srl_iterator, srl_model = (
        artefacts["readers"]["srl"],
        artefacts["iterators"]["srl"],
        artefacts["models"]["srl"],
    )
    if IS_MTL:
        if to_finetune == "coref":
            ft_model = coref_model.train()
            metric_key = "coref_f1"
        elif to_finetune == "srl":
            ft_model = srl_model.train()
            ft_model._tagger_srl.finetuning = True
            metric_key = "f1-measure-overall"
        else:
            raise ValueError("Invalid value for parameter `to_finetune`.")
        if not IS_BIG_MODEL:
            ft_model = ft_model.to(GPU)
    else:
        if to_finetune == "coref":
            ft_model = coref_model = coref_model.train().to(GPU)
            srl_model = srl_model.eval()
            metric_key = "coref_f1"
        elif to_finetune == "srl":
            coref_model = coref_model.eval()
            ft_model = srl_model = srl_model.train().to(GPU)
            ft_model._tagger_srl.finetuning = True
            metric_key = "f1-measure-overall"
        else:
            raise ValueError("Invalid value for parameter `to_finetune`.")

    best_state_dict = ft_model.state_dict()
    optimizer = get_optimizer(ft_model, to_finetune)
    doc_keys = list(wiki_articles.keys())
    coref_ft_instances = get_coref_instances(wiki_articles, coref_reader, coref_model)
    srl_ft_instances = get_srl_instances(wiki_articles, srl_reader, srl_model)
    dev_instances = get_dev_instances(to_finetune, finetune_dataset, ft_model)
    dev_iterator = coref_iterator if to_finetune == "coref" else srl_iterator
    best_metric_val, best_state_dict = 0, None
    historical_rewards = deque(maxlen=batch_size * 100)
    for epoch in range(epochs):
        shuffle(doc_keys)
        num_iters = math.ceil(len(doc_keys) / batch_size)
        for num_iter in range(num_iters):
            if num_iter % 1 == 0:
                ft_model = ft_model.eval()
                if IS_BIG_MODEL:
                    ft_model = ft_model.to(GPU)
                metrics, _ = evaluate(
                    ft_model,
                    dev_instances,
                    to_finetune,
                    dev_iterator,
                    cuda_device=GPU.index,
                )
                if IS_BIG_MODEL:
                    ft_model = ft_model.to(CPU)
                ft_model = ft_model.train()
                logger.info("{}: {}".format(metric_key, metrics[metric_key]))
                if metrics[metric_key] > best_metric_val:
                    logger.info("New best! Woohoo!")
                    best_metric_val = metrics[metric_key]
                    best_state_dict = copy.deepcopy(ft_model.state_dict())
                    torch.save(
                        best_state_dict,
                        os.path.join(
                            args.serialization_dir,
                            "best_finetuned_dgi_{}_{}.th".format(
                                to_finetune, finetune_dataset
                            ),
                        ),
                    )
                else:
                    ft_model.load_state_dict(copy.deepcopy(best_state_dict))
            start_time = time.time()
            rewards, log_likelihoods = [], []
            optimizer.zero_grad()
            no_verb_ctr = 0
            start_idx = num_iter * batch_size
            end_idx = min((num_iter + 1) * batch_size, len(doc_keys))
            coref_output = get_logits(
                [coref_ft_instances[key] for key in doc_keys[start_idx:end_idx]],
                coref_iterator,
                coref_model,
                task="coref",
                sample=(to_finetune == "coref"),
            )
            srl_output = get_logits(
                [srl_ft_instances[key] for key in doc_keys[start_idx:end_idx]],
                srl_iterator,
                srl_model,
                task="srl",
                sample=(to_finetune == "srl"),
            )
            combined_output = combine_outputs(coref_output, srl_output)
            assert len(combined_output) == end_idx - start_idx
            for dockey, co in combined_output.items():
                span_embeddings = (
                    torch.cat(
                        [
                            co["endpoint_span_embeddings"],
                            co["attended_span_embeddings"],
                        ],
                        dim=-1,
                    )
                    .squeeze()
                    .cpu()
                    .detach()
                    .numpy()
                )
                try:
                    nodes, edges, _ = json2graph(co)
                    graph, _ = Graph.build_graph(
                        dockey,
                        {"nodes": nodes, "edges": edges},
                        max_span_width=config["max_span_width"],
                        all_spans=co["all_spans"].cpu().detach().numpy()[0],
                        span_embeddings=span_embeddings,
                        original_text=co["document"],
                    )
                    if len(graph.nodes) == 0:
                        logging.info("Graph has 0 nodes. Skip fine-tuning...")
                        continue
                    reward = get_reward(clfs, graph)
                    logger.info(reward)
                    rewards.append(reward)
                    historical_rewards.append(reward)
                except NameError:
                    no_verb_ctr += 1
                    reward = torch.Tensor([-1.0]).item()
                    rewards.append(reward)
                    historical_rewards.append(reward)
                if to_finetune == "coref":
                    log_likelihoods.append(co["coref_log_likelihood"])
                else:
                    log_likelihoods.append(co["srl_log_likelihood"])
            baseline = (
                sum(historical_rewards) / len(historical_rewards)
                if len(historical_rewards) > batch_size
                else 0
            )
            loss = sum(
                -1.0 * (reward - baseline) * log_likelihood
                for reward, log_likelihood in zip(rewards, log_likelihoods)
            ) / (end_idx - start_idx)
            loss.backward()
            optimizer.step()
            logger.info(
                "Epoch {}({}/{}): no verbs: {}, loss = {:.2f}, avg. reward = {:.2f}, \
                baseline = {:.2f}, time: {:.2f}s".format(
                    epoch + 1,
                    num_iter + 1,
                    num_iters,
                    no_verb_ctr,
                    loss.item(),
                    sum(rewards) / len(rewards),
                    baseline,
                    time.time() - start_time,
                )
            )


def init_args():
    parser = argparse.ArgumentParser(
        description="Finetune pre-trained resolvers with policy gradient"
    )
    parser.add_argument("config_path", help="Configuration file")
    parser.add_argument(
        "reward_type", help="Type of reward model to load", choices=["sep", "com"]
    )
    parser.add_argument(
        "gencoder_type", help="Type of graph encoder", choices=["GCN", "GAT", "TAGCN"]
    )
    parser.add_argument("ft_task", help="Task to finetune", choices=["coref", "srl"])
    parser.add_argument(
        "ft_data",
        help="Dataset to finetune on",
        choices=[
            "conll12",
            "preco",
            "phrase_detectives_g",
            "phrase_detectives_w",
            "wikicoref",
            "winobias",
            "conll05",
            "ewt",
        ],
    )
    parser.add_argument(
        "serialization_dir",
        help="Path of the pre-trained model (finetuned model will be saved here)",
    )
    return parser.parse_args()


def init_single_models(serialization_dir, ft_task, device):
    best_models_dir = {
        "coref": "./models/single/coref_conll_base",
        "srl": "./models/single/srl_conll_base",
    }
    if ft_task == "coref":
        coref_reader, coref_iterator, coref_model = init_model(
            serialization_dir, device, task="coref"
        )
        srl_reader, srl_iterator, srl_model = init_model(
            best_models_dir["srl"], device, task="srl"
        )
    elif ft_task == "srl":
        srl_reader, srl_iterator, srl_model = init_model(
            serialization_dir, device, task="srl"
        )
        coref_reader, coref_iterator, coref_model = init_model(
            best_models_dir["coref"], device, task="coref"
        )
    else:
        raise ValueError("Invalid ft_task.")
    return {
        "readers": {"coref": coref_reader, "srl": srl_reader},
        "iterators": {"coref": coref_iterator, "srl": srl_iterator},
        "models": {"coref": coref_model, "srl": srl_model},
    }


if __name__ == "__main__":
    nlp = spacy.load("en")
    # use this only for debugging
    # torch.autograd.set_detect_anomaly(True)
    args = init_args()
    config = set_config(args.config_path)
    BDG.set_device(CPU)
    DGI.set_device(CPU)

    wiki_articles = read_wiki_articles(config["wiki_articles_path"])
    IS_BIG_MODEL = False
    IS_MTL = "coref" in args.serialization_dir and "srl" in args.serialization_dir
    if IS_MTL:
        # mtl models
        IS_BIG_MODEL = (
            False
            if "bert" in args.config_path
            and any(
                enc in args.config_path for enc in ["tiny", "mini", "small", "medium"]
            )
            else True
        )
        artefacts = init_pretrained_model(args.serialization_dir, CPU)
    else:
        # single models
        artefacts = init_single_models(args.serialization_dir, args.ft_task, CPU)

    # since we have already fixed the reward type, we don't take this from
    # the command-line arguments
    clfs = init_reward_models(
        config["reward_serialization_dir"],
        args.ft_task,
        args.reward_type,
        args.gencoder_type,
    )
    pg_loop(
        wiki_articles,
        artefacts,
        clfs,
        to_finetune=args.ft_task,
        finetune_dataset=args.ft_data,
    )
