import json
import os
import math

import torch
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import move_to_device

from transformers import RobertaTokenizer, RobertaModel

from hmtl.tasks import Task
from hmtl.common import create_and_set_iterators
from hmtl.dataset_readers import CorefConllReader, SrlReader

# variables which will be instantiated by init_span_embedder
combined_json, reader, iterator, model = None, None, None, None
roberta_tokenizer, roberta_model = None, None
include_bert = False


def get_map_loc():
    """
    Chooses device for tensors to be mapped based on the
    presence/absence of GPUs on a machine
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def init_model(serialization_dir, device, epoch=-1, task="coref"):
    """
    Load the best model from the `serialization_dir`
    If `epoch` is not -1, this method loads the model weights after the i-th epoch.
    """
    vocab = Vocabulary.from_files(os.path.join(serialization_dir, "./vocabulary"))
    params = Params.from_file(os.path.join(serialization_dir, "./config.json"))

    def get_reader_params(task_name):
        task_name = "task_{}".format(task_name)
        reader_params = params.get(task_name)["data_params"].get(
            "validation_dataset_reader", None
        )
        if reader_params is None:
            reader_params = params.pop(task_name)["data_params"]["dataset_reader"]
        return reader_params

    reader_params = get_reader_params(task)
    if task == "coref":
        reader_params.pop("type")
        reader = CorefConllReader.from_params(reader_params)
        iterator_params = params.pop("iterators")["iterator_coref"]
        iterator = DataIterator.from_params(iterator_params)
        best_model = (
            "best_coref.th" if epoch == -1 else "model_state_{}.th".format(epoch)
        )
    elif task == "srl":
        reader_params.pop("type")
        reader = SrlReader.from_params(reader_params)
        iterator_params = params.pop("iterators")["iterator_srl"]
        iterator = DataIterator.from_params(iterator_params)
        best_model = "best_srl.th" if epoch == -1 else "model_state_{}.th".format(epoch)
    else:
        raise ValueError("Undefined task.")

    model_params = params.pop("model")
    model = Model.from_params(
        vocab=vocab, params=model_params.duplicate(), regularizer=None
    )
    state = torch.load(os.path.join(serialization_dir, best_model), map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    return reader, iterator, model


def get_span_encoding(key, zero_span_rep=None):
    """
    Input: document key
    Output: all possible span tuples and their encodings
    """
    instance = reader.text_to_instance(combined_json[key]["sentences"])
    instance.index_fields(model.vocab)
    generator = iterator(instances=[instance])
    batch = next(generator)
    if type(get_map_loc().index) == int:
        batch = move_to_device(batch, get_map_loc().index)
    if zero_span_rep is not None:  # for debugging
        assert (
            zero_span_rep % 2 == 0
        ), "zero_span_rep must be even as it corresponds to concat(endpoint, attended)"
        shape = list(batch["spans"].shape)
        shape[-1] = int(zero_span_rep / 2)
        zeros = torch.zeros(shape)
        return {
            "original_text": batch["metadata"][0]["original_text"],
            "all_spans": batch["spans"],
            "endpoint_span_embeddings": zeros,
            "attended_span_embeddings": zeros,
            "roberta_embeddings": zeros,
        }
    output = model.forward(tensor_batch=batch, task_name="coref", for_training=False)
    reps = {
        "original_text": batch["metadata"][0]["original_text"],
        "all_spans": output["all_spans"],
        "endpoint_span_embeddings": output["endpoint_span_embeddings"],
        "attended_span_embeddings": output["attended_span_embeddings"],
    }
    if include_bert:
        reps["roberta_embeddings"] = get_bert_reps(
            combined_json[key]["sentences"], output["all_spans"][0]
        )
    return reps


def get_bert_reps(sentences, all_spans):
    reps = []
    input_ids = roberta_tokenizer.encode(
        [token for sentence in sentences for token in sentence],
        add_special_tokens=False,
    )
    if len(input_ids) > 512:
        to_fill = math.ceil(len(input_ids) / 512) * 512 - len(input_ids)
        input_ids = input_ids + [roberta_tokenizer.pad_token_id] * to_fill
        input_ids = torch.tensor(input_ids).view(-1, 512)
    else:
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        last_hidden_states = roberta_model(input_ids)[0]
        last_hidden_states = last_hidden_states.view(1, -1, 768)
    for (start, end) in all_spans:
        pooled, _ = torch.max(last_hidden_states[0][start : end + 1], dim=0)
        reps.append(pooled)
    return torch.stack(reps).unsqueeze(0)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_span_embedder(serialization_dir, combined_json_path):
    global combined_json, reader, iterator, model
    global roberta_tokenizer, roberta_model

    combined_json = read_json(os.path.join(serialization_dir, combined_json_path))
    reader, iterator, model = init_model(serialization_dir, get_map_loc())
    model.eval()
    if include_bert:
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = RobertaModel.from_pretrained("roberta-base")


def init_pretrained_model(serialization_dir, device):
    coref_reader, coref_iterator, coref_model = init_model(
        serialization_dir, device, task="coref"
    )
    srl_reader, srl_iterator, srl_model = init_model(
        serialization_dir, device, task="srl"
    )
    return {
        "readers": {"coref": coref_reader, "srl": srl_reader},
        "iterators": {"coref": coref_iterator, "srl": srl_iterator},
        "models": {"coref": coref_model, "srl": srl_model},
    }
