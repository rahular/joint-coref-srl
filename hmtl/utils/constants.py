import json
import numpy as np


_CONFIG = None


def r(d):
    return {w: i for i, w in enumerate(d)}


def set_config(path):
    global _CONFIG
    if _CONFIG:
        raise ValueError("Configuration already set.")
    else:
        with open(path, "r") as f:
            _CONFIG = json.load(f)
    return _CONFIG


def get_config():
    if not _CONFIG:
        raise ValueError("Configuration not set.")
    else:
        return _CONFIG


class K:
    perturbation_types = [
        "srl_change_label",
        "srl_move_arg",
        "srl_split_spans",
        "srl_merge_spans",
        "srl_change_boundary",
        "srl_add_arg",
        "srl_drop_arg",
        "coref_add_ant",
        "coref_drop_ant",
    ]
    perturbation_ratio = np.array(
        [
            29.3,
            4.5,
            10.6,
            14.7,
            18.0,
            7.4,
            11.0,  # from He et al. (2017)
            15.0,
            15.0,  # chosen arbitrarily
        ],
        dtype=float,
    )

    id2node_type = [
        "ROOT",
        "V",
        "ARG0",
        "ARG1",
        "ARG2",
        "ARG3",
        "ARG4",
        "ARG5",
        "ARGA",
        "ARGM-MOD",
        "ARGM-TMP",
        "ARGM-LOC",
        "ARGM-MNR",
        "ARGM-CAU",
        "ARGM-ADV",
        "ARGM-DIS",
        "ARGM-NEG",
        "ARGM-PRP",
        "ARGM-DIR",
        "ARGM-ADJ",
        "ARGM-PRD",
        "ARGM-PNC",
        "ARGM-EXT",
        "ARGM-COM",
        "ARGM-GOL",
        "ARGM-REC",
        "ARGM-PRP",
        "ARGM-DSP",
        "ARGM-SLC",
        "ARGM-LVB",
        "ARGM-PRR",
        "ARGM-PRX",
    ]

    node_type2id = r(id2node_type)
    id2edge_type = ["srl", "cor", "root"]
    edge_type2id = r(id2edge_type)

    id2arg_label = [
        "ARG0",
        "ARG1",
        "ARG2",
        "ARG3",
        "ARGM-ADV",
        "ARGM-DIR",
        "ARGM-LOC",
        "ARGM-MNR",
        "ARGM-PNC",
        "ARGM-TMP",
    ]
    arg_label2id = r(id2arg_label)
    arg_confusion = np.array(
        [  # from He et al. (2017). columns=pred, rows=gold
            [0, 55, 11, 13, 4, 0, 0, 0, 0, 0],
            [78, 0, 46, 0, 0, 22, 11, 10, 25, 14],
            [11, 23, 0, 48, 15, 56, 33, 41, 25, 0],
            [3, 2, 2, 0, 4, 0, 0, 0, 25, 14],
            [0, 0, 0, 4, 0, 0, 15, 29, 25, 36],
            [0, 0, 5, 4, 0, 0, 11, 2, 0, 0],
            [5, 9, 12, 0, 4, 0, 0, 10, 0, 14],
            [3, 0, 12, 26, 33, 0, 0, 0, 0, 21],
            [0, 3, 5, 4, 0, 11, 4, 2, 0, 0],
            [0, 8, 5, 0, 41, 11, 26, 6, 0, 0],
        ],
        dtype=float,
    )
