import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.
    
    Args:
        pos_scores: (batch_size,) scores for positive items
        neg_scores: (batch_size,) scores for negative items
    
    Returns:
        Scalar loss
    """
    # Loss = -log(sigmoid(pos - neg))
    # = log(1 + exp(-(pos - neg)))
    # = softplus(neg - pos)
    loss = F.softplus(neg_scores - pos_scores).mean()
    return loss


def binary_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Binary cross-entropy loss with logits.
    
    Args:
        logits: (batch_size,) predicted logits
        labels: (batch_size,) binary labels (0 or 1)
    
    Returns:
        Scalar loss
    """
    return F.binary_cross_entropy_with_logits(logits, labels.float())


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task GNN training.
    """
    
    def __init__(
        self,
        lambda_place: float = 1.0,
        lambda_friend: float = 0.5,
        lambda_attend: float = 0.3
    ):
        super().__init__()
        self.lambda_place = lambda_place
        self.lambda_friend = lambda_friend
        self.lambda_attend = lambda_attend
    
    def forward(
        self,
        loss_place: torch.Tensor,
        loss_friend: torch.Tensor,
        loss_attend: torch.Tensor
    ) -> tuple:
        """
        Combine losses with weights.
        
        Returns:
            total_loss: Weighted sum
            loss_dict: Individual loss values for logging
        """
        total_loss = (
            self.lambda_place * loss_place +
            self.lambda_friend * loss_friend +
            self.lambda_attend * loss_attend
        )
        
        loss_dict = {
            'loss_place': loss_place.item(),
            'loss_friend': loss_friend.item(),
            'loss_attend': loss_attend.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict

