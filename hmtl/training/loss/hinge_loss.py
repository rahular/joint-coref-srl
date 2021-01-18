import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        hinge_loss = self.margin - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return torch.sum(hinge_loss)
