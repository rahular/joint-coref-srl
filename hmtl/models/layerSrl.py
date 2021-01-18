# coding: utf-8

import os
import sys
import logging
from typing import Dict
from overrides import overrides

import torch

from transformers import AutoModel
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.span_extractors import (
    SelfAttentiveSpanExtractor,
    EndpointSpanExtractor,
)
from allennlp.modules import FeedForward
from allennlp.models.crf_tagger import CrfTagger

from hmtl.modules.text_field_embedders import ShortcutConnectTextFieldEmbedder
from hmtl.models import SrlCustom, SrlCustomBert

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("srl_custom")
class LayerSrl(Model):
    """
    A class that implement the one task of HMTL model: SRL `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`.
    
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

        super(LayerSrl, self).__init__(vocab=vocab, regularizer=regularizer)

        if params.get("bert_model", None):
            logger.info("Loading BERT as embedder for Srl")
            _bert_model_name = params.pop("bert_model")
            self.bert_model = AutoModel.from_pretrained(_bert_model_name)
            self.bert_model._name = _bert_model_name
            tagger_srl = SrlCustomBert(
                vocab=vocab,
                bert_model=self.bert_model,
                embedding_dropout=params.pop_float("embedding_dropout"),
                label_smoothing=params.pop_float("label_smoothing"),
            )
            self._tagger_srl = tagger_srl

        else:
            # Base text Field Embedder
            text_field_embedder_params = params.pop("text_field_embedder")
            text_field_embedder = BasicTextFieldEmbedder.from_params(
                vocab=vocab, params=text_field_embedder_params
            )
            self._text_field_embedder = text_field_embedder

            ##############
            # SRL Stuffs
            ##############
            srl_params = params.pop("srl")

            # Encoder
            encoder_params = srl_params.pop("encoder")
            encoder = Seq2SeqEncoder.from_params(encoder_params)
            self._encoder = encoder

            # Secondary Encoder
            secondary_encoder_params = srl_params.pop("secondary_encoder")
            secondary_encoder_params[
                "input_size"
            ] = self._encoder.get_output_dim() + srl_params.get("binary_feature_dim")
            secondary_encoder = Seq2SeqEncoder.from_params(secondary_encoder_params)
            self._secondary_encoder = secondary_encoder

            # Tagger: SRL
            init_params = srl_params.pop("initializer", None)
            initializer = (
                InitializerApplicator.from_params(init_params)
                if init_params is not None
                else InitializerApplicator()
            )

            tagger_srl = SrlCustom(
                vocab=vocab,
                text_field_embedder=self._text_field_embedder,
                encoder=self._encoder,
                secondary_encoder=self._secondary_encoder,
                binary_feature_dim=srl_params.pop_int("binary_feature_dim", 100),
                label_smoothing=srl_params.pop_float("label_smoothing", 0.1),
                initializer=initializer,
            )
            self._tagger_srl = tagger_srl

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
        tagger = getattr(self, "_tagger_%s" % task_name)
        tensor_batch["sample"] = sample
        return tagger.forward(**tensor_batch)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor], task_name: str = "srl"):
        tagger = getattr(self, "_tagger_%s" % task_name)
        return tagger.decode(output_dict)

    @overrides
    def get_metrics(
        self, task_name: str = "srl", reset: bool = False, full: bool = False
    ) -> Dict[str, float]:
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset=reset)

    @classmethod
    def from_params(
        cls,
        vocab: Vocabulary,
        params: Params,
        regularizer: RegularizerApplicator,
        **kwargs
    ) -> "LayerSrl":
        return cls(vocab=vocab, params=params, regularizer=regularizer)
