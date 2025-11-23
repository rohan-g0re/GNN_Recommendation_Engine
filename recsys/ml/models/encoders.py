import torch
import torch.nn as nn
from recsys.config.model_config import ModelConfig
from recsys.config.constants import (
    N_CITIES, N_NEIGHBORHOODS_PER_CITY, N_PRICE_BANDS, N_TIME_SLOTS,
    C_COARSE, C_FINE, C_VIBE, MAX_NEIGHBORHOODS_PER_USER
)

class UserEncoder(nn.Module):
    """
    Encodes raw user features to D_MODEL dimensional embedding.
    
    Input shape: (batch_size, D_user_raw=148)
    Output shape: (batch_size, D_MODEL)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers for categorical features
        self.city_embed = nn.Embedding(
            num_embeddings=N_CITIES,
            embedding_dim=config.CITY_EMBED_DIM
        )
        self.neighborhood_embed = nn.Embedding(
            num_embeddings=N_NEIGHBORHOODS_PER_CITY,
            embedding_dim=config.NEIGHBORHOOD_EMBED_DIM
        )
        
        # Calculate input dimension to MLP
        # Embeddings: city + neighborhood
        embed_dim = config.CITY_EMBED_DIM + config.NEIGHBORHOOD_EMBED_DIM
        # Continuous features: cat_pref + fine_pref + vibe_pref + area_freqs + behavior_stats
        continuous_dim = C_COARSE + C_FINE + C_VIBE + MAX_NEIGHBORHOODS_PER_USER + 5
        mlp_input_dim = embed_dim + continuous_dim
        
        # MLP to project to D_MODEL
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, config.D_MODEL * 2),
            nn.LayerNorm(config.D_MODEL * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.D_MODEL * 2, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 148) raw user features
                [0:1] = city_id
                [1:2] = neighborhood_id
                [2:8] = cat_pref
                [8:108] = fine_pref
                [108:138] = vibe_pref
                [138:143] = area_freqs
                [143:148] = behavior_stats
        
        Returns:
            (batch_size, D_MODEL) encoded user embeddings
        """
        # Extract categorical IDs
        city_ids = x[:, 0].long()
        neighborhood_ids = x[:, 1].long()
        
        # Embed categoricals
        city_emb = self.city_embed(city_ids)  # (batch, city_embed_dim)
        neigh_emb = self.neighborhood_embed(neighborhood_ids)  # (batch, neigh_embed_dim)
        
        # Extract continuous features
        continuous = x[:, 2:]  # (batch, 146)
        
        # Concatenate all
        combined = torch.cat([city_emb, neigh_emb, continuous], dim=1)
        
        # Project to D_MODEL
        out = self.mlp(combined)
        
        return out


class PlaceEncoder(nn.Module):
    """
    Encodes raw place features to D_MODEL dimensional embedding.
    
    Input shape: (batch_size, D_place_raw=114)
    Output shape: (batch_size, D_MODEL)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.city_embed = nn.Embedding(N_CITIES, config.CITY_EMBED_DIM)
        self.neighborhood_embed = nn.Embedding(N_NEIGHBORHOODS_PER_CITY, config.NEIGHBORHOOD_EMBED_DIM)
        self.price_embed = nn.Embedding(N_PRICE_BANDS, config.PRICE_EMBED_DIM)
        self.time_slot_embed = nn.Embedding(N_TIME_SLOTS, config.TIME_SLOT_EMBED_DIM)
        
        # Calculate MLP input dimension
        embed_dim = (config.CITY_EMBED_DIM + config.NEIGHBORHOOD_EMBED_DIM + 
                     config.PRICE_EMBED_DIM + config.TIME_SLOT_EMBED_DIM)
        # Continuous: category_one_hot + fine_tag_vector + popularity (4)
        continuous_dim = C_COARSE + C_FINE + 4
        mlp_input_dim = embed_dim + continuous_dim
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, config.D_MODEL * 2),
            nn.LayerNorm(config.D_MODEL * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.D_MODEL * 2, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 114) raw place features
                [0:1] = city_id
                [1:2] = neighborhood_id
                [2:8] = category_one_hot
                [8:108] = fine_tag_vector
                [108:109] = price_band
                [109:110] = typical_time_slot
                [110:114] = popularity metrics
        
        Returns:
            (batch_size, D_MODEL) encoded place embeddings
        """
        # Extract and embed categoricals
        city_ids = x[:, 0].long()
        neighborhood_ids = x[:, 1].long()
        price_bands = x[:, 108].long()
        time_slots = x[:, 109].long()
        
        city_emb = self.city_embed(city_ids)
        neigh_emb = self.neighborhood_embed(neighborhood_ids)
        price_emb = self.price_embed(price_bands)
        time_emb = self.time_slot_embed(time_slots)
        
        # Extract continuous features
        category_onehot = x[:, 2:8]
        fine_tags = x[:, 8:108]
        popularity = x[:, 110:114]
        
        continuous = torch.cat([category_onehot, fine_tags, popularity], dim=1)
        
        # Concatenate all
        combined = torch.cat([
            city_emb, neigh_emb, price_emb, time_emb, continuous
        ], dim=1)
        
        # Project to D_MODEL
        out = self.mlp(combined)
        
        return out

