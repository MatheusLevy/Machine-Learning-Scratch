import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        # Calculate binary cross entropy
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Calculate the modulating factor
        pt = torch.exp(-BCE_loss)
        focal_term = (1 - pt) ** self.gamma

        # Combine focal term and alpha
        focal_loss = self.alpha * focal_term * BCE_loss

        return torch.mean(focal_loss)