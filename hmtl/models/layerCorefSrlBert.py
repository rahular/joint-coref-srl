# coding: utf-8

import logging
from typing import Dict

import torch
from transformers import AutoModel
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.span_extractors import (
    SelfAttentiveSpanExtractor,
    EndpointSpanExtractor,
)
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from overrides import overrides

from hmtl.models import CoreferenceCustom, SrlCustomBert

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("coref_srl_bert")
class LayerCorefSrlBert(Model):
    """
    A class that implement two tasks of HMTL model: EMD (CRF Tagger) and Coref (Lee et al., 2017).
    
    Parameters
    ----------
    vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
    params: ``allennlp.common.Params``, required
        Configuration parameters for the multi-task model.
    regularizer: ``allennlp.nn.RegularizerApplicator``, optional (default = None)
        A reguralizer to apply to the model's layers.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        params: Params,
        regularizer: RegularizerApplicator = None,
    ):
        super(LayerCorefSrlBert, self).__init__(vocab=vocab, regularizer=regularizer)

        srl_params = params.pop("srl")
        coref_params = params.pop("coref")

        # Base text Field Embedder
        _bert_model_name = srl_params.pop("bert_model")
        bert_model = AutoModel.from_pretrained(_bert_model_name)
        bert_model._name = _bert_model_name
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder_params["token_embedders"]["tokens"][
            "model_name"
        ] = bert_model
        text_field_embedder = BasicTextFieldEmbedder.from_params(
            vocab=vocab, params=text_field_embedder_params
        )
        self._text_field_embedder = text_field_embedder

        ############
        # SRL Stuffs
        ############
        tagger_srl = SrlCustomBert(
            vocab=vocab,
            bert_model=bert_model,
            label_smoothing=params.pop_float("label_smoothing", 0.1),
        )
        self._tagger_srl = tagger_srl

        ##############
        # Coref Stuffs
        ##############

        # Encoder
        encoder_coref_params = coref_params.pop("encoder")
        encoder_coref = Seq2SeqEncoder.from_params(encoder_coref_params)
        self._encoder_coref = encoder_coref

        # Tagger: Coreference
        tagger_coref_params = coref_params.pop("tagger")
        eval_on_gold_mentions = tagger_coref_params.pop_bool(
            "eval_on_gold_mentions", False
        )
        init_params = tagger_coref_params.pop("initializer", None)
        initializer = (
            InitializerApplicator.from_params(init_params)
            if init_params is not None
            else InitializerApplicator()
        )

        # Span embedders
        self._endpoint_span_extractor = EndpointSpanExtractor(
            self._encoder_coref.get_output_dim(),
            combination="x,y",
            num_width_embeddings=tagger_coref_params.get("max_span_width", 10),
            span_width_embedding_dim=tagger_coref_params.get("feature_size", 20),
            bucket_widths=False,
        )
        input_embedding_size = self._text_field_embedder.get_output_dim()
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=input_embedding_size
        )

        tagger_coref = CoreferenceCustom(
            vocab=vocab,
            text_field_embedder=self._text_field_embedder,
            context_layer=self._encoder_coref,
            mention_feedforward=FeedForward.from_params(
                tagger_coref_params.pop("mention_feedforward")
            ),
            antecedent_feedforward=FeedForward.from_params(
                tagger_coref_params.pop("antecedent_feedforward")
            ),
            feature_size=tagger_coref_params.pop_int("feature_size"),
            max_span_width=tagger_coref_params.pop_int("max_span_width"),
            spans_per_word=tagger_coref_params.pop_float("spans_per_word"),
            max_antecedents=tagger_coref_params.pop_int("max_antecedents"),
            lexical_dropout=tagger_coref_params.pop_float("lexical_dropout", 0.2),
            initializer=initializer,
            eval_on_gold_mentions=eval_on_gold_mentions,
        )
        self._tagger_coref = tagger_coref

        if eval_on_gold_mentions:
            self._tagger_coref._eval_on_gold_mentions = True

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(
        self,
        tensor_batch,
        for_training: bool = False,
        task_name: str = "srl",
        sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Special case for forward: for coreference, we can use gold mentions to predict the clusters
        during evaluation (not during training).
        """

        tagger = getattr(self, "_tagger_" + task_name)

        if task_name == "coref" and tagger._eval_on_gold_mentions:
            if for_training:
                tagger._use_gold_mentions = False
            else:
                tagger._use_gold_mentions = True

        tensor_batch["sample"] = sample
        return tagger.forward(**tensor_batch)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor], task_name: str = "srl"):
        tagger = getattr(self, "_tagger_" + task_name)
        return tagger.decode(output_dict)

    @overrides
    def get_metrics(
        self, task_name: str = "srl", reset: bool = False, full: bool = False
    ) -> Dict[str, float]:

        task_tagger = getattr(self, "_tagger_" + task_name)
        if full and task_name == "coref":
            return task_tagger.get_metrics(reset=reset, full=full)
        else:
            return task_tagger.get_metrics(reset=reset)

    @classmethod
    def from_params(
        cls,
        vocab: Vocabulary,
        params: Params,
        regularizer: RegularizerApplicator,
        **kwargs
    ) -> "LayerCorefSrlBert":
        return cls(vocab=vocab, params=params, regularizer=regularizer)
