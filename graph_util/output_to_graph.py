import json
import ntpath
import os
import sys
import logging

from itertools import combinations
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_end_index(current_index, tags_list):
    length = 0
    for tag in tags_list[current_index + 1 :]:
        if tag[0] == "I":
            length += 1
        else:
            break
    return length


def get_node_type(tag):
    type_tag = tag.lstrip("B-").lstrip("R-").lstrip("C-")
    return type_tag


def get_doc_indices(doc, pattern):
    """ This is required since the doc and sentence 
    indices are not always aligned. We return the first 
    matched index. While this does not hurt the model, 
    it is not the best way to do it.
    """
    for i in range(len(doc)):
        if doc[i] == pattern[0] and doc[i : i + len(pattern)] == pattern:
            return (i, i + len(pattern) - 1)


def extract_srl_nodes_and_edges(
    all_srl_tags, document_text, all_sentences, add_root_node=True
):
    """ Input is a list of srl tags for sentences
    each of which has 1 or more dictionaries per
    verb with the verb with the verb and tags.

    Output is a list of nodes each of which is a dict
    and a list of edges which are node id pairs.
    """
    nodes, edges, verb_node_ids = [], [], []
    node_id = 0
    for srl_tags_per_sentence, sentence in zip(all_srl_tags, all_sentences):
        if not srl_tags_per_sentence:
            continue

        for verb_dict in srl_tags_per_sentence:
            tags = verb_dict["tags"]
            vdict_node_ids = []
            for tag_index, tag in enumerate(tags):
                if not tag.startswith("B"):
                    continue

                constituent_len = get_end_index(tag_index, tags)
                type = get_node_type(tag)
                text = sentence[tag_index : tag_index + constituent_len + 1]
                doc_indices = get_doc_indices(document_text, text)
                nodes.append(
                    {
                        "node_id": node_id,
                        "type": type,
                        "doc_indices": doc_indices,
                        "text": text,
                    }
                )
                vdict_node_ids.append(node_id)
                if type == "V":
                    verb_id = node_id
                    verb_node_ids.append(verb_id)
                node_id += 1

            vdict_edges = [
                {"nids": [verb_id, n_id], "type": "srl"}
                for n_id in vdict_node_ids
                if n_id != verb_id
            ]
            edges.extend(vdict_edges)

    # add root node
    if add_root_node:
        if nodes:
            nodes.append(
                {"node_id": node_id, "type": "ROOT", "doc_indices": (-1, -1), "text": None}
            )
            vdict_edges = [
                {"nids": [node_id, verb_id], "type": "root"}
                for verb_id in verb_node_ids
            ]
            edges.extend(vdict_edges)
    return nodes, edges


def get_overlapping_span(span, doc_indices):
    best_left, best_right, best_span_len = 10e9, -10e9, 10e9
    s, e = span
    for (ds, de) in doc_indices:
        if s - ds >= 0 and e - de <= 0 and best_span_len > de - ds:
            best_left, best_right, best_span_len = ds, de, de - ds
    return (best_left, best_right)


def get_coref_edges(srl_nodes, coref_clusters):
    doc_node_indices = [node["doc_indices"] for node in srl_nodes]
    doc_node_ids = [node["node_id"] for node in srl_nodes]
    span_in_doc_counter = 0
    new_edges = []
    # logger.info(sorted(doc_node_indices, key=lambda x: x[0]))
    for cluster in coref_clusters:
        matching_span_node_ids = set()
        for span in cluster:
            span_in_doc_counter += 1
            overlapping_span = get_overlapping_span(span, doc_node_indices)
            # logger.info(span, overlapping_span)
            if overlapping_span != (10e9, -10e9):
                span_index_in_doc_indices = doc_node_indices.index(overlapping_span)
                span_node_id = doc_node_ids[span_index_in_doc_indices]
                matching_span_node_ids.add(span_node_id)
        if len(matching_span_node_ids) > 1:
            new_edges.extend(combinations(matching_span_node_ids, 2))
    if new_edges is not None:
        new_edges = [{"nids": n, "type": "cor"} for n in new_edges]
    return new_edges, span_in_doc_counter


def json2graph(graph_json):
    metadata = dict()
    document_text = graph_json["document"]
    all_srl_tags = graph_json["srl_tags"]
    corref_clusters = graph_json["clusters"]
    sentences = graph_json["sentences"]
    nodes, edges = extract_srl_nodes_and_edges(all_srl_tags, document_text, sentences)
    coref_edges, _ = get_coref_edges(nodes, corref_clusters[0])
    edges.extend(coref_edges)
    metadata["empty_doc"] = True if not nodes else False
    metadata["coref_edges"] = True if coref_edges else False
    metadata["num_sent"] = len(sentences)
    # logger.info(json.dumps({'nodes': nodes, 'edges': edges}, indent=2))
    return nodes, edges, metadata


def dump_graph(graphs, input_path):
    head, tail = ntpath.split(input_path)
    with open(os.path.join(head, "graphs_{}".format(tail)), "w", encoding="utf-8") as f:
        json.dump(graphs, f)


if __name__ == "__main__":
    # with open('./combined_outputs.json') as f:
    #     combined_outputs = json.load(f)
    # co = combined_outputs[1]

    # nodes, edges, metadata = json2graph(co)

    input_filepath = sys.argv[1]
    data_file = open(input_filepath, "r", encoding="UTF-8").read()
    empty_docs_count = 0
    corref_edges_count = 0
    graph_dataset = {}

    output_sets = json.loads(data_file)

    for doc_key, output_set in tqdm(
        output_sets.items(), unit=" graphs", desc="Reading {}".format(data_file)
    ):
        nodes, edges, metadata = json2graph(output_set)
        graph_dataset[doc_key] = {"nodes": nodes, "edges": edges, "metadata": metadata}
        empty_docs_count += int(metadata["empty_doc"])
        corref_edges_count += int(metadata["corref_edges"])
    print(empty_docs_count, " : empty docs count")
    print(corref_edges_count, " : corref_edges_count")
    # dump graph to file
    dump_graph(graph_dataset, input_filepath)
