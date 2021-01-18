import torch

from overrides import overrides
from sklearn.metrics import f1_score

from allennlp.training.metrics.metric import Metric


class F1(Metric):
    def __init__(self) -> None:
        self.f1 = 0
        self.total = 0

    def clamp(self, logits):
        logits[logits >= 0] = 1
        logits[logits < 0] = -1
        return logits

    @overrides
    def __call__(self, logits, labels):
        logits = list(self.unwrap_to_tensors(logits))[0]
        logits = logits.detach()
        labels = labels.detach()
        logits = self.clamp(logits)
        self.f1 += f1_score(
            labels.cpu().numpy(), logits.cpu().numpy(), average="binary"
        )
        self.total += 1

    @overrides
    def get_metric(self, reset: bool = False):
        f1_value = self.f1 / self.total if self.total > 0 else 0
        if reset:
            self.reset()
        return f1_value

    @overrides
    def reset(self):
        self.f1 = 0
        self.total = 0
