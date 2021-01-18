import os
import json
import argparse
import torch

from tqdm import tqdm

from allennlp.data import Instance, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.data.iterators import DataIterator
from allennlp.common.params import Params

from hmtl.utils import set_config
from hmtl.dataset_readers import (
    CorefConllReader,
    CorefPrecoReader,
    CorefWikicorefReader,
    CorefWinobiasReader,
    SrlReader,
    SrlReader05,
    SrlWikibankReader,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args):
    vocab = Vocabulary.from_files(os.path.join(args.serialization_dir, "./vocabulary"))
    params = Params.from_file(os.path.join(args.serialization_dir, "./config.json"))
    print(params.get("iterators"))
    iterator_params = params.pop("iterators")["iterator_{}".format(args.task)]
    iterator = DataIterator.from_params(iterator_params)
    if args.model_type == "pt":
        best_model = "best_{}.th".format(args.task)
    else:
        best_model = "best_finetuned_dgi_{}_{}.th".format(args.task, args.data)

    model_params = params.pop("model")
    model = Model.from_params(
        vocab=vocab, params=model_params.duplicate(), regularizer=None
    )
    state = torch.load(os.path.join(args.serialization_dir, best_model), map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    return iterator, model


def get_instances(args, config, model):
    reader_classes = {
        "coref": {
            "conll12": CorefConllReader,
            "preco": CorefPrecoReader,
            "wikicoref": CorefWikicorefReader,
            "winobias": CorefWinobiasReader,
            "phrase_detectives_g": CorefConllReader,
            "phrase_detectives_w": CorefConllReader,
        },
        "srl": {
            "conll12": SrlReader,
            "conll05_w": SrlReader05,
            "conll05_b": SrlReader05,
            "wikibank": SrlWikibankReader,
            "ewt": SrlWikibankReader,
        },
    }
    params = Params.from_file(os.path.join(args.serialization_dir, "./config.json"))
    task_name = "task_{}".format(args.task)
    reader_params = params.get(task_name)["data_params"].get(
        "validation_dataset_reader", None
    )
    if reader_params is None:
        reader_params = params.pop(task_name)["data_params"]["dataset_reader"]
    reader_params.pop("type")
    reader = reader_classes[args.task][args.data].from_params(reader_params)
    instances = []
    for instance in reader._read(
        config["{}_{}_test_data_path".format(args.task, args.data)]
    ):
        instance.index_fields(model.vocab)
        instances.append(instance)
    return instances


def get_outputs(args, instances, iterator, model):
    generator = iterator(instances=instances, num_epochs=1)
    outputs = []
    for batch in tqdm(generator):
        if device.type != "cpu":
            batch = util.move_to_device(batch, device.index)
        output = model.forward(
            tensor_batch=batch, task_name=args.task, for_training=False, sample=False
        )
        output = model.decode(output, task_name=args.task)
        if args.task == "coref":
            outputs.append(
                {
                    "document": output["document"],
                    "gold_clusters": output["gold_clusters"],
                    "predicted_clusters": output["clusters"],
                }
            )
        elif args.task == "srl":
            words, gtags, ptags = output["words"], output["gold_tags"], output["tags"]
            assert len(words) == len(gtags) == len(ptags)
            for w, gt, pt in zip(words, gtags, ptags):
                outputs.append(
                    {"words": w, "gold_tags": gt, "predicted_tags": pt}
                )
    return outputs


def process_coref(data):
    text = []
    for d in data:
        doc = [str(token) for token in d["document"][0]]    # just to be safe
        gcs = d["gold_clusters"][0]
        pcs = d["predicted_clusters"][0]
        gcs = (
            [sorted(cluster, key=lambda x: (x[0], x[1])) for cluster in gcs]
            if len(gcs) > 0
            else []
        )
        pcs = (
            [sorted(cluster, key=lambda x: (x[0], x[1])) for cluster in pcs]
            if len(pcs) > 0
            else []
        )

        # assume the first mention in every gold cluster to be the antecedent
        # every other mention in that cluster will be the coreferent mention
        # search for every coreferent mention in the predicted clusters
        triples = []
        not_found = []
        for gc in gcs:
            gant = gc[0]
            for gmention in gc[1:]:
                found = True
                for pc in pcs:
                    pant = pc[0]
                    if gmention in pc[1:]:
                        triples.append(
                            "\t".join(
                                [
                                    " ".join(doc[gmention[0] : gmention[1] + 1]),
                                    " ".join(doc[gant[0] : gant[1] + 1]),
                                    " ".join(doc[pant[0] : pant[1] + 1]),
                                ]
                            )
                        )
                        found = True
                        break
                if not found:
                    not_found.append(gmention)
        text.extend(
            [
                "'''\n{}\nMissing mentions: [{}]\n'''\n".format(
                    doc, ", ".join(not_found)
                ),
                "\n".join(triples),
                "\n\n",
            ]
        )
    return text


def process_srl(data):
    text = []
    for d in data:
        words = d["words"]
        gtags = d["gold_tags"]
        ptags = d["predicted_tags"]
        doc = " ".join(words)
        text.append("'''\n{}\n'''\n".format(doc))
        for w, g, p in zip(words, gtags, ptags):
            text.append("\t".join([w, g, p]) + "\n")
        text.append("\n")
    return text


def write(args, outputs):
    file_name = "./outputs/{}/{}_{}_{}.txt".format(
        args.model_type, args.task, args.serialization_dir.split("/")[-1], args.data
    )
    with open(file_name, "w", encoding="utf-8") as f:
        if args.task == "coref":
            f.writelines(process_coref(outputs))
        elif args.task == "srl":
            f.writelines(process_srl(outputs))


def init_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Fine-tuning configuration file.",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--serialization_dir",
        required=True,
        help="Directory where the model is saved.",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        help="Name of the task to predict.",
        choices=["coref", "srl"],
        type=str,
    )
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="Name of the dataset to load.",
        choices=[
            "conll12",
            "preco",
            "phrase_detectives_g",
            "phrase_detectives_w",
            "wikicoref",
            "winobias",
            "conll05_w",
            "conll05_b",
            "ewt",
        ],
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_type",
        required=True,
        help="Model type to load (pre-trained or fine-tuned).",
        choices=["pt", "ft"],
        type=str,
    )
    return parser.parse_args()


def main():
    args = init_args()
    config = set_config(args.config_path)
    iterator, model = init_model(args)
    instances = get_instances(args, config, model)
    outputs = get_outputs(args, instances, iterator, model)
    write(args, outputs)


if __name__ == "__main__":
    main()
