
import torch
from torch import nn
import torch.nn.functional as F

class classification_probe(nn.Module):
    def __init__(self, n_classes):
        super(classification_probe, self).__init__()
        self._n_classes = n_classes
        self.fc = nn.Linear(n_classes, n_classes)

    def forward(self, x, y):
        logits = self.fc(x.detach())
        xe = F.nll_loss(F.log_softmax(logits, dim=-1), y) #  logits is equal to x when minmizing the xe.
        # xe = torch.mean(xe)
        acc = torch.mean((torch.argmax(logits, dim=1) == y).float())
        return xe, acc