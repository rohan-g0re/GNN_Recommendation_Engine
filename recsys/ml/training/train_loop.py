import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from recsys.ml.models.losses import bpr_loss, binary_cross_entropy_loss, CombinedLoss
from recsys.config.model_config import ModelConfig


class GNNTrainer:
    """
    Trainer for GNN recommendation model.
    """
    
    def __init__(
        self,
        user_encoder,
        place_encoder,
        backbone,
        place_head,
        friend_head,
        place_ctx_encoder,
        friend_ctx_encoder,
        graph,
        config: ModelConfig,
        device: str = 'cuda'
    ):
        self.user_encoder = user_encoder.to(device)
        self.place_encoder = place_encoder.to(device)
        self.backbone = backbone.to(device)
        self.place_head = place_head.to(device)
        self.friend_head = friend_head.to(device)
        self.place_ctx_encoder = place_ctx_encoder.to(device)
        self.friend_ctx_encoder = friend_ctx_encoder.to(device)
        self.graph = graph.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        params = list(self.user_encoder.parameters()) + \
                 list(self.place_encoder.parameters()) + \
                 list(self.backbone.parameters()) + \
                 list(self.place_head.parameters()) + \
                 list(self.friend_head.parameters()) + \
                 list(self.place_ctx_encoder.parameters()) + \
                 list(self.friend_ctx_encoder.parameters())
        
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.loss_fn = CombinedLoss(
            lambda_place=config.LAMBDA_PLACE,
            lambda_friend=config.LAMBDA_FRIEND,
            lambda_attend=config.LAMBDA_ATTEND
        )
    
    def train_epoch(self, place_loader, friend_loader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.user_encoder.train()
        self.place_encoder.train()
        self.backbone.train()
        self.place_head.train()
        self.friend_head.train()
        
        total_loss_place = 0.0
        total_loss_friend = 0.0
        total_loss_attend = 0.0
        num_batches = 0
        
        # Get graph embeddings (recompute each epoch as model updates)
        x_user = self.user_encoder(self.graph['user'].x)
        x_place = self.place_encoder(self.graph['place'].x)
        x_dict = {'user': x_user, 'place': x_place}
        
        z_dict = self.backbone(
            x_dict,
            self.graph.edge_index_dict,
            self.graph.edge_attr_dict
        )
        
        # Train on place recommendations
        for batch in place_loader:
            user_indices = batch['user_idx'].to(self.device)
            pos_place_indices = batch['pos_place_idx'].to(self.device)
            neg_place_indices = batch['neg_place_idx'].to(self.device)
            
            z_users = z_dict['user'][user_indices]
            z_pos_places = z_dict['place'][pos_place_indices]
            z_neg_places = z_dict['place'][neg_place_indices]
            
            # Dummy context (zeros)
            ctx = torch.zeros(len(user_indices), self.config.D_CTX_PLACE).to(self.device)
            
            pos_scores = self.place_head(z_users, z_pos_places, ctx)
            neg_scores = self.place_head(z_users, z_neg_places, ctx)
            
            loss_place = bpr_loss(pos_scores, neg_scores)
            
            self.optimizer.zero_grad()
            loss_place.backward()
            self.optimizer.step()
            
            total_loss_place += loss_place.item()
            num_batches += 1
        
        # Train on friend compatibility
        for batch in friend_loader:
            user_u_indices = batch['user_u_idx'].to(self.device)
            user_v_indices = batch['user_v_idx'].to(self.device)
            labels_compat = batch['label_compat'].to(self.device).float()
            labels_attend = batch['label_attend'].to(self.device).float()
            
            z_user_u = z_dict['user'][user_u_indices]
            z_user_v = z_dict['user'][user_v_indices]
            
            # Dummy context
            ctx = torch.zeros(len(user_u_indices), self.config.D_CTX_FRIEND).to(self.device)
            
            compat_logits, attend_prob = self.friend_head(z_user_u, z_user_v, ctx)
            
            loss_friend = binary_cross_entropy_loss(compat_logits, labels_compat)
            loss_attend = binary_cross_entropy_loss(
                torch.logit(attend_prob + 1e-8), labels_attend
            )
            
            total_loss = loss_friend + loss_attend
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_loss_friend += loss_friend.item()
            total_loss_attend += loss_attend.item()
        
        return {
            'loss_place': total_loss_place / max(num_batches, 1),
            'loss_friend': total_loss_friend / max(len(friend_loader), 1),
            'loss_attend': total_loss_attend / max(len(friend_loader), 1)
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'user_encoder': self.user_encoder.state_dict(),
            'place_encoder': self.place_encoder.state_dict(),
            'backbone': self.backbone.state_dict(),
            'place_head': self.place_head.state_dict(),
            'friend_head': self.friend_head.state_dict(),
            'place_ctx_encoder': self.place_ctx_encoder.state_dict(),
            'friend_ctx_encoder': self.friend_ctx_encoder.state_dict(),
            'config': self.config,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

