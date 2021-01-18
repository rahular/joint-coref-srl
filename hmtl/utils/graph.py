import random
import sys
from collections import defaultdict
from typing import Optional, Dict, Tuple
from uuid import uuid4

import numpy as np
from scipy.special import softmax

from hmtl.utils import K

MAX_PERTURB_ATTEMPTS = (
    10  # maximum number of attempts to apply the same perturbation if it fails
)


class Node:
    def __init__(self, node_id, node_type, doc_indices, text):
        self.node_id = node_id
        self.type = node_type
        self.doc_indices: Optional[Tuple[int]] = None if doc_indices is None else tuple(
            doc_indices
        )
        self.text = text
        self.perturbed = False

        self.links = []
        self.link_types = []

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return self.node_id

    def __str__(self):
        return "node_id: {} node_type: {} doc_incides: {} text: {} links: {} link_types: {}".format(
            self.node_id,
            self.type,
            self.doc_indices,
            self.text,
            self.links,
            self.link_types,
        )

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(
            Node.__name__, self.node_id, self.type, self.doc_indices, self.text
        )


class Graph:
    # static class variable to compute sample probabilities
    _node_freqs = defaultdict(int)

    def __init__(
        self,
        graph_id=None,
        metadata=None,
        max_span_width=10,
        all_spans=None,
        span_embeddings=None,
        original_text=None,
    ):
        self.id = graph_id or str(uuid4())
        self.nodes: Dict[int, Node] = dict()
        self.root = None
        self.max_span_width = max_span_width
        self.all_spans = [tuple(span) for span in all_spans]
        self.span_embeddings = span_embeddings
        self.span_embeddings_by_span = (
            None
            if self.all_spans is None or self.span_embeddings is None
            else dict(zip(self.all_spans, self.span_embeddings))
        )
        self.original_text = original_text
        self.node_spans = []
        self.node_texts = []
        self.metadata = metadata or {}
        self.perturbations_applied = []

    def get_edges(self):
        edges = []
        edge_types = []
        for src_id, node in self.nodes.items():
            for dest_node, link_type in zip(node.links, node.link_types):
                edges.append((src_id, dest_node.node_id))
                edge_types.append(link_type)
        return edges, edge_types

    def __len__(self):
        return len(self.nodes)

    def add_node(self, node):
        idx = node.node_id
        if idx in self.nodes:
            raise ValueError("`node_id` must be unique.")
        else:
            self.nodes[idx] = node

        if node.type == "ROOT":
            if self.root:
                raise ValueError("There can be only one ROOT per document")
            else:
                self.root = node
                self.root.text = ["$ROOT$"]
                self.node_spans.append(None)
                self.node_texts.append("ROOT")

        else:
            # add span to all spans for sampling later
            self.node_spans.append(node.doc_indices)
            self.node_texts.append(node.text)

    def del_node(self, node_id_del):
        node_del = self.nodes[node_id_del]

        # find nodes connected to node chosen for deletion, remove edges
        linked_src_nodes = [
            node.node_id for node in self.nodes.values() if node_del in node.links
        ]
        linked_dest_nodes = [
            node.node_id for node in self.nodes.values() if node in node_del.links
        ]
        for src_node_id in linked_src_nodes:
            # print(src_node_id, "src_node_id")
            # print([link.node_id for link in self.nodes[src_node_id].links],
            #      'link.node_id for link in self.src_node_id.links')
            self.del_edge(src=src_node_id, dest=node_del.node_id)
        for dest_node_id in linked_dest_nodes:
            # print(dest_node_id, "dest_node_id")
            # print([link.node_id for link in self.nodes[sampled_node_id].links],
            #      'link.node_id for link in self.nodes[sampled_node_id].links')
            self.del_edge(src=node_del.node_id, dest=dest_node_id)

        # delete the node from the list
        del self.nodes[node_id_del]

        # shift ids to fill gap
        copy_nodes = dict()
        for node_key, node in self.nodes.items():
            if node.node_id > node_id_del:
                node.node_id -= 1
            copy_nodes[node.node_id] = node
        self.nodes = copy_nodes

        # remove span and text from graph
        self.node_spans.remove(node_del.doc_indices)
        self.node_texts.remove(node_del.text)

    def add_edge(self, src, dest, edge_type):
        # if src or dest nodes does not exist
        if src not in self.nodes or dest not in self.nodes:
            raise ValueError("Nodes should exist before linking.")
        # elif type != 'cor':
        src_node = self.nodes[src]
        dest_node = self.nodes[dest]
        src_node.links.append(dest_node)
        src_node.link_types.append(edge_type)

        Graph._node_freqs[src_node.type] += 1
        Graph._node_freqs[dest_node.type] += 1

    def del_edge(self, src, dest):
        # if src or dest nodes does not exist
        if src not in self.nodes or dest not in self.nodes:
            raise ValueError("Nodes should exist before deleting.")
        # elif type != 'cor':
        src_node = self.nodes[src]
        dest_node = self.nodes[dest]

        # get dest index in src_node's edges to be able to del
        dest_index = src_node.links.index(dest_node)

        src_node.links.pop(dest_index)
        src_node.link_types.pop(dest_index)

    def is_empty(self):
        return not self.nodes

    def topsort(self):
        stack = []
        visited = set()

        def recursive_helper(helper_node):
            visited.add(helper_node)
            for neighbor in helper_node.links:
                if neighbor not in visited:
                    # if neighbor is a verb (head node), start a new recursion
                    if neighbor.type == "V":
                        stack.append(["$START$", "MT"])
                    recursive_helper(neighbor)
                    if neighbor.type == "V":
                        stack.append(["$END$", "MT"])
            stack.append([" ".join(helper_node.text), helper_node.type])

        if not self.is_empty():
            # start with the ROOT node
            stack.append(["$START$", "MT"])
            recursive_helper(self.root)
            stack.append(["$END$", "MT"])

            for node in self.nodes.values():
                # continue with head nodes
                if node not in visited and len(node.links) > 0:
                    stack.append(["$START$", "MT"])
                    recursive_helper(node)
                    stack.append(["$END$", "MT"])

            for node in self.nodes.values():
                # end with singletons
                if node not in visited:
                    stack.append(["$START$", "MT"])
                    recursive_helper(node)
                    stack.append(["$END$", "MT"])
        return stack

    @staticmethod
    def _make_probs(types):
        frequencies = np.array([Graph._node_freqs[t] for t in types])
        return softmax(frequencies / np.max(frequencies))

    def sample_node(self, predicate=None) -> Optional[Node]:
        sampling_pool = [
            node
            for node in self.nodes.values()
            if node.type not in ("V", "ROOT") and (predicate is None or predicate(node))
        ]
        if not sampling_pool:
            return None
        sampling_types = [node.type for node in sampling_pool]
        return np.random.choice(sampling_pool, p=self._make_probs(sampling_types))

    def set_span(self, node, span):
        self.node_spans.remove(node.doc_indices)
        self.node_texts.remove(node.text)
        node.doc_indices = span
        node.text = self.original_text[span[0] : span[1] + 1]
        self.node_spans.append(node.doc_indices)
        self.node_texts.append(node.text)

    def perturb(self, perturbation_probabilities, no_perturb_actions=1):
        """ apply number of perturbations equal to the number of sentences
        input: probability of each perturbation type in K.perturbation_types given we decided to perturb"""

        assert len(K.perturbation_types) == len(
            perturbation_probabilities
        ), "List of perturbations must be of equal size to list of proportions"

        if self.is_empty():
            raise ValueError("Empty Graph")

        for perturbation in np.random.choice(
            K.perturbation_types, no_perturb_actions, p=perturbation_probabilities
        ):
            try:
                perturbation_function = getattr(self, perturbation)
            except AttributeError as e:
                raise ValueError(
                    "Perturbation '{}' is not supported".format(perturbation)
                ) from e
            for _ in range(
                MAX_PERTURB_ATTEMPTS
            ):  # retry perturbation until success or until max attempts
                result = None
                try:
                    result = perturbation_function()
                except ValueError as e:
                    print("Failed applying perturbation {}:\n{}".format(perturbation, e), file=sys.stderr)
                if result:  # succeeded to apply perturbation, move on
                    self.perturbations_applied.append(perturbation + " " + result)
                    break

    def srl_change_label(self):
        # sample gold label to change based on the overall probability of error on that label
        existing_labels = {node.type for node in self.nodes.values()}
        if not existing_labels.intersection(K.id2arg_label):  # There are no nodes
            return False
        labels, source_label_prob = zip(
            *[
                (label, weight)
                for label, weight in zip(K.id2arg_label, K.arg_confusion.sum(axis=0))
                if label in existing_labels
            ]
        )
        source_label_prob = np.array(source_label_prob, dtype=float)
        source_label_prob /= source_label_prob.sum()
        sampled_source_label = np.random.choice(labels, p=source_label_prob)
        # sample replacement label conditioned on the source label
        target_label_prob = K.arg_confusion[K.arg_label2id[sampled_source_label]]
        target_label_prob /= target_label_prob.sum()
        sampled_target_label = np.random.choice(K.id2arg_label, p=target_label_prob)
        # sample node id uniformly among nodes with the source label
        sampling_pool = [
            node.node_id
            for node in self.nodes.values()
            if node.type == sampled_source_label
        ]
        sampled_node_id = np.random.choice(sampling_pool)
        # replace label
        sampled_node = self.nodes[sampled_node_id]
        sampled_node.type = sampled_target_label
        sampled_node.perturbed = True
        return "{} type to {}".format(sampled_node_id, sampled_target_label)

    def srl_move_arg(self):
        # sample argument to move
        sampled_node = self.sample_node(
            lambda node: node.doc_indices[1] - node.doc_indices[0] < self.max_span_width
        )
        if sampled_node is None:  # no nodes found
            return False
        # sample target position among spans of the same length that do not intersect with the source one
        span_embedding = self.span_embeddings_by_span[sampled_node.doc_indices]
        all_non_intersecting_same_len_spans = [
            span
            for span in self.all_spans
            if set(span) != set(sampled_node.doc_indices)
            and span[1] - span[0]
            == sampled_node.doc_indices[1] - sampled_node.doc_indices[0]
            and (
                span[1] < sampled_node.doc_indices[0]
                or span[0] > sampled_node.doc_indices[1]
            )
        ]
        if not all_non_intersecting_same_len_spans:
            return False
        embeddings = [
            self.span_embeddings_by_span[span]
            for span in all_non_intersecting_same_len_spans
        ]
        similarities = [
            np.dot(embedding, span_embedding) / np.linalg.norm(embedding)
            for embedding in embeddings
        ]
        probabilities = np.exp(similarities - np.max(similarities))
        probabilities /= np.sum(probabilities)
        i = np.random.choice(len(all_non_intersecting_same_len_spans), p=probabilities)
        sampled_span = all_non_intersecting_same_len_spans[i]
        # set new span and text
        self.set_span(sampled_node, sampled_span)
        sampled_node.perturbed = True
        return "{} span to {}".format(sampled_node.node_id, sampled_span)

    def srl_split_spans(self):
        # sample argument to split
        sampled_node = self.sample_node(
            lambda node: node.doc_indices[1] - node.doc_indices[0] > 2
        )
        if sampled_node is None:  # no nodes found or span too short
            return False
        # sample splitting point uniformly: first half ends at this point
        first_end = np.random.randint(
            sampled_node.doc_indices[0] + 1, sampled_node.doc_indices[1]
        )
        second_start = first_end + 1
        if sampled_node.doc_indices[1] - sampled_node.doc_indices[0] >= 3:
            second_start += np.random.randint(
                0, 1
            )  # sometimes separate the splits by one word
        if (
            (sampled_node.doc_indices[0], first_end) not in self.span_embeddings_by_span
            or (second_start, sampled_node.doc_indices[1])
            not in self.span_embeddings_by_span
        ):
            return False  # For some reason one of the spans has no embedding (maybe crosses sentences)
        # create new node of the same argument type for second half
        new_node_id = max(self.nodes) + 1
        new_node = Node(
            new_node_id,
            sampled_node.type,
            (second_start, sampled_node.doc_indices[1]),
            sampled_node.text[second_start - sampled_node.doc_indices[0] :],
        )
        self.add_node(new_node)
        # trim old node to first half
        self.set_span(sampled_node, (sampled_node.doc_indices[0], first_end))
        # copy link from Verb nodes
        linked_src_verb_node_ids = [
            node.node_id
            for node in self.nodes.values()
            if node.type == "V" and sampled_node in node.links
        ]
        for verb_node_id in linked_src_verb_node_ids:
            self.add_edge(src=verb_node_id, dest=new_node_id, edge_type="srl")
        sampled_node.perturbed = new_node.perturbed = True
        return "{} to {} and {} ({})".format(
            sampled_node.node_id,
            sampled_node.doc_indices,
            new_node.doc_indices,
            new_node_id,
        )

    def srl_merge_spans(self):
        # find pairs of argument nodes with at most one separating token
        argument_nodes = [
            node for node in self.nodes.values() if node.type not in ("V", "ROOT")
        ]
        adjacent_pairs = [
            (node1, node2)
            for node1 in argument_nodes
            for node2 in argument_nodes
            if 1 <= node2.doc_indices[0] - node1.doc_indices[1] <= 2
            and node1.doc_indices[1] - node1.doc_indices[0] < self.max_span_width
            and (node1.doc_indices[0], node2.doc_indices[1])
            in self.span_embeddings_by_span
        ]
        if not adjacent_pairs:
            return False
        # sample a pair uniformly among them
        sampled_node1, sampled_node2 = random.sample(adjacent_pairs, 1)[0]
        # merge them by expanding the first and dropping the second
        self.set_span(
            sampled_node1, (sampled_node1.doc_indices[0], sampled_node2.doc_indices[1])
        )
        # drop second node including all edges
        self.del_node(sampled_node2.node_id)
        sampled_node1.perturbed = True
        return "{} and {} to {}".format(
            sampled_node1.node_id, sampled_node2.node_id, sampled_node1.doc_indices
        )

    def srl_change_boundary(self):
        # sample argument to change the boundaries of
        sampled_node = self.sample_node()
        if sampled_node is None:  # no nodes found
            return False
        # sample shifted boundaries
        start = random.randint(
            max((0, sampled_node.doc_indices[0] - 5)),
            min(sampled_node.doc_indices[1] + 1, len(self.original_text)),
        )
        end = random.randint(
            start,
            min(
                (
                    sampled_node.doc_indices[1] + 5,
                    start + self.max_span_width,
                    len(self.original_text) + 1,
                )
            ),
        )
        if (start, end) == sampled_node.doc_indices or (
            start,
            end,
        ) not in self.span_embeddings_by_span:
            return False  # not changed
        self.set_span(sampled_node, (start, end))
        sampled_node.perturbed = True
        return "{} to {}".format(sampled_node.node_id, sampled_node.doc_indices)

    def srl_add_arg(self):
        # sample node type
        sampling_types = [t for t in K.id2node_type if t not in ("V", "ROOT")]
        sampled_ntype = np.random.choice(
            sampling_types, p=self._make_probs(sampling_types)
        )
        # sample node text
        doc_span_lengths = set(list(span[1] - span[0] for span in self.node_spans if span))
        #doc_span_lengths_all = list(span[1] - span[0] for span in self.node_spans if span)
        #print(doc_span_lengths, "doc_span_lengths")
        #print(doc_span_lengths_all, 'doc_span_lengths_all')
        sampled_span_length = np.random.choice(
            list(doc_span_lengths.intersection(range(self.max_span_width)))
        )
        #print(sampled_span_length, "sampled_span_length")
        #print(len(self.node_spans), len(self.node_texts), "len(self.node_spans), len(self.node_texts)")

        all_spans_filtered = [
            (span, text)
            for span, text in zip(self.node_spans, self.node_texts)
            if span
        ]
        #print(all_spans_filtered, "all_spans_filtered")



        all_spans_filtered_by_len = [
            (span_text[0], span_text[1])
            for span_text in all_spans_filtered
            if span_text[0][1] - span_text[0][0] == sampled_span_length
        ]
        #print(all_spans_filtered_by_len, "all_spans_filtered_by_len")

        sampled_span, sampled_text = random.sample(all_spans_filtered_by_len, 1)[0]
        # get node id
        node_id = max(self.nodes) + 1
        node = Node(node_id, sampled_ntype, sampled_span, sampled_text)
        self.add_node(node)
        # sample Verb node to connect new node to
        sampling_pool_predicates = [
            node.node_id for node in self.nodes.values() if node.type == "V"
        ]
        sampled_verb_node_id = np.random.choice(sampling_pool_predicates)
        self.add_edge(src=sampled_verb_node_id, dest=node_id, edge_type="srl")
        node.perturbed = True
        return "{} ({})".format(node.doc_indices, node.node_id)

    def srl_drop_arg(self):
        # sample argument to drop
        sampled_node = self.sample_node()
        if sampled_node is None:  # no nodes found
            return False
        # delete node
        self.del_node(sampled_node.node_id)
        return "{} ({})".format(sampled_node.doc_indices, sampled_node.node_id)

    def coref_add_ant(self):
        # sample node ids
        sampling_pool = [
            node.node_id
            for node in self.nodes.values()
            if node.type not in ("V", "ROOT")
        ]
        if len(sampling_pool) < 2:
            return False
        sampled_node_id_1, sampled_node_id_2 = random.sample(sampling_pool, 2)
        self.add_edge(src=sampled_node_id_1, dest=sampled_node_id_2, edge_type="cor")
        self.nodes[sampled_node_id_1].perturbed = self.nodes[
            sampled_node_id_2
        ].perturbed = True
        return "{} and {}".format(sampled_node_id_1, sampled_node_id_2)

    def coref_drop_ant(self):
        # sample node ids
        all_srl_arg_nodes = [
            node.node_id
            for node in self.nodes.values()
            if node.type not in ("V", "ROOT")
        ]
        all_coref_edges = []
        for srl_arg_node_id in all_srl_arg_nodes:
            srl_arg_node = self.nodes[srl_arg_node_id]
            coref_edges_indices = [
                idx
                for idx, edge_type in enumerate(srl_arg_node.link_types)
                if edge_type == "cor"
            ]
            for coref_edge_idx in coref_edges_indices:
                all_coref_edges.append((srl_arg_node_id, coref_edge_idx))
        if not all_coref_edges:
            return False
        src_id, dest_index = random.sample(all_coref_edges, 1)[0]
        src_node = self.nodes[src_id]
        dest_id = src_node.links[dest_index].node_id
        self.del_edge(src_id, dest_id)
        dest_node = self.nodes[dest_id]
        src_node.perturbed = dest_node.perturbed = True
        return "{} and {}".format(src_id, dest_id)

    @staticmethod
    def handle_coref_chains(g, ling):
        mentions = set()
        for e in g["edges"]:
            if e["type"] == "cor":
                mentions.add(e["nids"][0])
                mentions.add(e["nids"][1])
        for node in ling:
            if node[0] in mentions:
                node.append(True)
            else:
                node.append(False)
        return ling

    @staticmethod
    def build_graph(
        dockey,
        g,
        max_span_width=10,
        all_spans=None,
        span_embeddings=None,
        original_text=None,
    ):
        graph = Graph(
            dockey,
            metadata=g.get("metadata"),
            max_span_width=max_span_width,
            all_spans=all_spans,
            span_embeddings=span_embeddings,
            original_text=original_text,
        )
        for n in g["nodes"]:
            node = Node(n["node_id"], n["type"], n["doc_indices"], n["text"])
            graph.add_node(node)
        for e in g["edges"]:
            graph.add_edge(e["nids"][0], e["nids"][1], e["type"])
        ling = graph.topsort()
        ling = Graph.handle_coref_chains(g, ling)
        return graph, ling

    def to_json(self):
        nodes = []
        edges = []
        for node in self.nodes.values():
            nodes.append(
                {
                    "node_id": node.node_id,
                    "type": node.type,
                    "doc_indices": node.doc_indices,
                    "text": node.text,
                    "perturbed": node.perturbed,
                }
            )
            for dest, etype in zip(node.links, node.link_types):
                edges.append({"nids": [node.node_id, dest.node_id], "type": etype})
        return {"nodes": nodes, "edges": edges}
