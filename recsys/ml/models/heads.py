import torch
import torch.nn as nn
from recsys.config.model_config import ModelConfig
from recsys.config.constants import N_CITIES, N_TIME_SLOTS, C_COARSE


class PlaceHead(nn.Module):
    """
    Scoring head for user-place recommendations.
    
    Takes user embedding, place embedding, and context.
    Outputs scalar relevance score.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimension: z_u + z_p + z_u*z_p + |z_u-z_p| + ctx
        # = D_MODEL + D_MODEL + D_MODEL + D_MODEL + D_CTX_PLACE
        # = 4 * D_MODEL + D_CTX_PLACE
        input_dim = 4 * config.D_MODEL + config.D_CTX_PLACE
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.PLACE_HEAD_HIDDEN:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.HEAD_DROPOUT)
            ])
            prev_dim = hidden_dim
        
        # Final scoring layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        z_user: torch.Tensor,
        z_place: torch.Tensor,
        ctx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_user: (batch_size, D_MODEL) user embeddings
            z_place: (batch_size, D_MODEL) place embeddings
            ctx: (batch_size, D_CTX_PLACE) context features
        
        Returns:
            (batch_size, 1) relevance scores
        """
        # Construct interaction features
        z_mul = z_user * z_place  # Element-wise product
        z_diff = torch.abs(z_user - z_place)  # Absolute difference
        
        # Concatenate all features
        features = torch.cat([z_user, z_place, z_mul, z_diff, ctx], dim=1)
        
        # Score
        score = self.mlp(features)  # (batch_size, 1)
        
        return score.squeeze(-1)  # (batch_size,)


class ContextEncoder(nn.Module):
    """
    Encodes context features for place/friend recommendations.
    """
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        
        # Embeddings for context
        self.city_embed = nn.Embedding(N_CITIES, 8)
        self.time_slot_embed = nn.Embedding(N_TIME_SLOTS, 8)
        
        # For optional desired category/tags (multi-hot input)
        # Input: C_COARSE + small MLP
        self.category_proj = nn.Linear(C_COARSE, output_dim // 2)
        
        # Combined projection
        self.proj = nn.Linear(8 + 8 + output_dim // 2, output_dim)
    
    def forward(
        self,
        city_id: torch.Tensor,
        time_slot: torch.Tensor,
        desired_categories: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            city_id: (batch_size,) city IDs
            time_slot: (batch_size,) time slot IDs
            desired_categories: (batch_size, C_COARSE) multi-hot desired categories
        
        Returns:
            (batch_size, output_dim) context vector
        """
        city_emb = self.city_embed(city_id)
        time_emb = self.time_slot_embed(time_slot)
        cat_emb = torch.relu(self.category_proj(desired_categories))
        
        combined = torch.cat([city_emb, time_emb, cat_emb], dim=1)
        ctx = self.proj(combined)
        
        return ctx


class FriendHead(nn.Module):
    """
    Scoring head for user-user compatibility.
    
    Outputs two scores:
    - Compatibility score (how well they match)
    - Attendance probability (likelihood to accept/attend)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimension
        input_dim = 4 * config.D_MODEL + config.D_CTX_FRIEND
        
        # Shared trunk
        trunk_layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.FRIEND_HEAD_HIDDEN[:-1]:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.HEAD_DROPOUT)
            ])
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Compatibility head (outputs logit)
        self.compat_head = nn.Sequential(
            nn.Linear(prev_dim, config.FRIEND_HEAD_HIDDEN[-1]),
            nn.ReLU(),
            nn.Linear(config.FRIEND_HEAD_HIDDEN[-1], 1)
        )
        
        # Attendance probability head (outputs logit, will apply sigmoid)
        self.attend_head = nn.Sequential(
            nn.Linear(prev_dim, config.FRIEND_HEAD_HIDDEN[-1]),
            nn.ReLU(),
            nn.Linear(config.FRIEND_HEAD_HIDDEN[-1], 1)
        )
    
    def forward(
        self,
        z_user_u: torch.Tensor,
        z_user_v: torch.Tensor,
        ctx: torch.Tensor
    ) -> tuple:
        """
        Args:
            z_user_u: (batch_size, D_MODEL) query user embeddings
            z_user_v: (batch_size, D_MODEL) candidate user embeddings
            ctx: (batch_size, D_CTX_FRIEND) context features
        
        Returns:
            compat_score: (batch_size,) compatibility scores (logits)
            attend_prob: (batch_size,) attendance probabilities (0-1)
        """
        # Construct interaction features
        z_mul = z_user_u * z_user_v
        z_diff = torch.abs(z_user_u - z_user_v)
        
        # Concatenate
        features = torch.cat([z_user_u, z_user_v, z_mul, z_diff, ctx], dim=1)
        
        # Shared representation
        shared = self.trunk(features)
        
        # Two heads
        compat_logits = self.compat_head(shared).squeeze(-1)  # (batch_size,)
        attend_logits = self.attend_head(shared).squeeze(-1)
        attend_prob = torch.sigmoid(attend_logits)  # (batch_size,)
        
        return compat_logits, attend_prob
    
    def compute_combined_score(
        self,
        compat_score: torch.Tensor,
        attend_prob: torch.Tensor,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Combine compatibility and attendance into final ranking score.
        
        Args:
            compat_score: Compatibility scores (logits or normalized)
            attend_prob: Attendance probabilities (0-1)
            alpha: Weight for compatibility (1-alpha for attendance)
        
        Returns:
            Combined scores
        """
        # Normalize compat_score to [0, 1] via sigmoid
        compat_norm = torch.sigmoid(compat_score)
        
        return alpha * compat_norm + (1 - alpha) * attend_prob

