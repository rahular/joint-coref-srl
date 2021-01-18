from typing import Dict
import logging
import pickle as pkl

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes


logger = logging.getLogger(__name__)


@DatasetReader.register("conll_lm")
class ConllLanguageModelingReader(DatasetReader):
    """
    Reads the Ontonotes data and converts it into a ``Dataset`` suitable for training a language model.
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    max_tokens : ``int``, optional (default=``None``)
        If this is ``None``, we will have each training instance be a single sentence.  If this is
        not ``None``, we will instead take all sentences, including their start and stop tokens,
        line them up, and split the tokens into groups of this number, for more efficient training
        of the language model.
    lazy : ``boolean``, optional (default=``False``)
        If this is ``True``, the data is loaded lazily
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or WordTokenizer()

        self._source_token_indexers = source_token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self._target_token_indexers = target_token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self._max_tokens = max_tokens

    def _normalize_word(self, word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def read_ontonotes(self, file_path):
        ontonotes_reader = Ontonotes()
        for doc in ontonotes_reader.dataset_document_iterator(file_path):
            yield [START_SYMBOL] + [
                word for sentence in doc for word in sentence.words
            ] + [
                END_SYMBOL
            ]  # List[str]

    # read from the `graphs_combined_dev.json` pickle
    @overrides
    def _read(self, file_path: str):
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        for key, values in data.items():
            document = [START_SYMBOL] + values["document"][0] + [END_SYMBOL]
            node_repr = values["node_repr"]
            # we need not tokenize here because Ontonotes sentences are already tokenized
            tokenized_doc = [Token(word) for word in document]
            if self._max_tokens is not None:
                num_tokens = self._max_tokens + 1
                tokenized_text = []
                for index in range(0, len(tokenized_doc) - num_tokens, num_tokens - 1):
                    tokenized_text.append(tokenized_doc[index : (index + num_tokens)])
            else:
                tokenized_text = [tokenized_doc]

            for text in tokenized_text:
                input_field = TextField(text[:-1], self._source_token_indexers)
                output_field = TextField(text[1:], self._target_token_indexers)
                yield Instance(
                    {
                        "source_tokens": input_field,
                        "target_tokens": output_field,
                        "metadata": MetadataField(
                            {"key": key, "node_repr": node_repr, "document": document}
                        ),
                    }
                )

    # read from Ontonotes file
    def _readO(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        for doc in self.read_ontonotes(file_path):
            # we need not tokenize here because Ontonotes sentences are already tokenized
            tokenized_doc = [Token(word) for word in doc]
            if self._max_tokens is not None:
                num_tokens = self._max_tokens + 1
                tokenized_text = []
                for index in range(0, len(tokenized_doc) - num_tokens, num_tokens - 1):
                    tokenized_text.append(tokenized_doc[index : (index + num_tokens)])
            else:
                tokenized_text = [tokenized_doc]

            for text in tokenized_text:
                input_field = TextField(text[:-1], self._source_token_indexers)
                output_field = TextField(text[1:], self._target_indexer)
                yield Instance(
                    {"source_tokens": input_field, "target_tokens": output_field}
                )

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        source_text = self._source_tokenizer.tokenize(sentence)
        target_text = self._target_tokenizer.tokenize(sentence)
        input_field = TextField(source_text[:-1], self._source_token_indexers)
        output_field = TextField(target_text[1:], self._target_token_indexers)
        return Instance({"source_tokens": input_field, "target_tokens": output_field})


if __name__ == "__main__":
    lm_reader = ConllLanguageModelingReader()
    for idx, instance in enumerate(
        lm_reader._read("../../models/coref_srl_conll/graphs/dev_node_reps.pkl")
    ):
        print(instance)
