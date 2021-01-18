import torch

from overrides import overrides

from allennlp.training.metrics.metric import Metric


class Accuracy(Metric):
    def __init__(self) -> None:
        self.correct = 0
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
        self.correct += torch.eq(logits.cpu(), labels.cpu()).sum().item()
        self.total += labels.size(0)

    @overrides
    def get_metric(self, reset: bool = False):
        accuracy_value = self.correct / self.total if self.total > 0 else 0
        if reset:
            self.reset()
        return accuracy_value

    @overrides
    def reset(self):
        self.correct = 0
        self.total = 0
