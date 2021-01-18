# coding: utf-8

import sys
import logging
import json
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from collections import defaultdict
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from hmtl.dataset_readers.dataset_utils import ACE

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("coref_wikicoref")
class CorefWikicorefReader(DatasetReader):
    """
    A dataset reader to read the coref clusters from an ACE dataset
    previously pre-procesed to fit the CoNLL-coreference format.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default = False)
        Whether or not the dataset should be loaded in lazy way. 
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        subset_size: int = sys.maxsize,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._to_yield = subset_size

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        data = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))

        for instance in data:
            if self._to_yield == 0:
                break
            sentences = instance["sentences"]
            clusters = [
                [tuple(span) for span in cluster] for cluster in instance["clusters"]
            ]
            yield self.text_to_instance(sentences, clusters)
            self._to_yield -= 1

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Instance:

        """
        # Parameters
        sentences : `List[List[str]]`, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.
        # Returns
        An `Instance` containing the following `Fields`:
            text : `TextField`
                The text of the full document.
            spans : `ListField[SpanField]`
                A ListField containing the spans represented as `SpanFields`
                with respect to the document text.
            span_labels : `SequenceLabelField`, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a `SequenceLabelField`
                 with respect to the `spans `ListField`.
        """
        flattened_sentences = [
            self._normalize_word(word) for sentence in sentences for word in sentence
        ]

        if self._wordpiece_modeling_tokenizer is not None:
            (
                flat_sentences_tokens,
                offsets,
            ) = self._wordpiece_modeling_tokenizer.intra_word_tokenize(
                flattened_sentences
            )
            flattened_sentences = [t.text for t in flat_sentences_tokens]
        else:
            flat_sentences_tokens = [Token(word) for word in flattened_sentences]

        text_field = TextField(flat_sentences_tokens, self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            if self._wordpiece_modeling_tokenizer is not None:
                for cluster in gold_clusters:
                    for mention_id, mention in enumerate(cluster):
                        start = offsets[mention[0]][0]
                        end = offsets[mention[1]][1]
                        cluster[mention_id] = (start, end)

            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(
                sentence, offset=sentence_offset, max_span_width=self._max_span_width
            ):
                if self._wordpiece_modeling_tokenizer is not None:
                    start = offsets[start][0]
                    end = offsets[end][1]

                    # `enumerate_spans` uses word-level width limit; here we apply it to wordpieces
                    # We have to do this check here because we use a span width embedding that has
                    # only `self._max_span_width` entries, and since we are doing wordpiece
                    # modeling, the span width embedding operates on wordpiece lengths. So a check
                    # here is necessary or else we wouldn't know how many entries there would be.
                    if end - start + 1 > self._max_span_width:
                        continue
                    # We also don't generate spans that contain special tokens
                    if (
                        start
                        < self._wordpiece_modeling_tokenizer.num_added_start_tokens
                    ):
                        continue
                    if (
                        end
                        >= len(flat_sentences_tokens)
                        - self._wordpiece_modeling_tokenizer.num_added_end_tokens
                    ):
                        continue

                if span_labels is not None:
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)

                spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)

        span_field = ListField(spans)

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": span_field,
            "metadata": metadata_field,
        }
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
