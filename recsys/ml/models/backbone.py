import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from typing import Dict
from recsys.config.model_config import ModelConfig


class EdgeAwareSAGEConv(nn.Module):
    """
    SAGEConv that incorporates edge attributes.
    """
    
    def __init__(self, in_channels, out_channels, edge_dim, aggr='mean'):
        super().__init__()
        self.sage = SAGEConv(in_channels, out_channels, aggr=aggr)
        # Project edge features to weights
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (tuple of tensors) source and target node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
        
        Returns:
            Updated node embeddings
        """
        # Get edge weights from edge attributes
        edge_weights = self.edge_mlp(edge_attr)  # (E, out_channels)
        
        # Mean over channel dimension to get scalar weights
        edge_weights_scalar = edge_weights.mean(dim=1)  # (E,)
        
        # Use as edge weights in SAGE aggregation
        return self.sage(x, edge_index, edge_weight=edge_weights_scalar)


class GraphRecBackbone(nn.Module):
    """
    Heterogeneous GNN backbone for user and place embeddings.
    
    Implements L layers of message passing over:
    - ('user', 'social', 'user') edges
    - ('user', 'interacts', 'place') edges
    - ('place', 'rev_interacts', 'user') edges
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.NUM_GNN_LAYERS
        
        # Build heterogeneous convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleDict()
        
        for layer_idx in range(self.num_layers):
            # Input dimension (first layer uses D_MODEL, rest use GNN_HIDDEN_DIM)
            in_dim = config.D_MODEL if layer_idx == 0 else config.GNN_HIDDEN_DIM
            out_dim = config.GNN_HIDDEN_DIM
            
            # Define convolutions for each relation type
            conv_dict = {
                # User-user social edges
                ('user', 'social', 'user'): EdgeAwareSAGEConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    edge_dim=3,  # D_edge_uu
                    aggr=config.GNN_AGGR
                ),
                
                # User-place interaction edges
                ('user', 'interacts', 'place'): EdgeAwareSAGEConv(
                    in_channels=(in_dim, in_dim),  # (user_dim, place_dim)
                    out_channels=out_dim,
                    edge_dim=12,  # D_edge_up
                    aggr=config.GNN_AGGR
                ),
                
                # Place-user reverse edges
                ('place', 'rev_interacts', 'user'): EdgeAwareSAGEConv(
                    in_channels=(in_dim, in_dim),  # (place_dim, user_dim)
                    out_channels=out_dim,
                    edge_dim=12,  # D_edge_up
                    aggr=config.GNN_AGGR
                ),
            }
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
            # Layer normalization for each node type
            self.norms[f'user_{layer_idx}'] = nn.LayerNorm(out_dim)
            self.norms[f'place_{layer_idx}'] = nn.LayerNorm(out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.GNN_DROPOUT)
        
        # Final projection to D_MODEL (if GNN_HIDDEN_DIM != D_MODEL)
        if config.GNN_HIDDEN_DIM != config.D_MODEL:
            self.final_proj_user = nn.Linear(config.GNN_HIDDEN_DIM, config.D_MODEL)
            self.final_proj_place = nn.Linear(config.GNN_HIDDEN_DIM, config.D_MODEL)
        else:
            self.final_proj_user = nn.Identity()
            self.final_proj_place = nn.Identity()
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        edge_attr_dict: Dict[tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: {'user': (N_user, D_MODEL), 'place': (N_place, D_MODEL)}
            edge_index_dict: {
                ('user', 'social', 'user'): (2, E_uu),
                ('user', 'interacts', 'place'): (2, E_up),
                ('place', 'rev_interacts', 'user'): (2, E_up)
            }
            edge_attr_dict: {
                ('user', 'social', 'user'): (E_uu, 3),
                ('user', 'interacts', 'place'): (E_up, 12),
                ('place', 'rev_interacts', 'user'): (E_up, 12)
            }
        
        Returns:
            {'user': (N_user, D_MODEL), 'place': (N_place, D_MODEL)}
        """
        # Initialize with encoder outputs
        h_dict = x_dict
        
        # Apply GNN layers
        for layer_idx in range(self.num_layers):
            # Heterogeneous convolution
            h_dict_new = self.convs[layer_idx](
                h_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict
            )
            
            # Apply activation, normalization, and dropout
            for node_type in ['user', 'place']:
                h = h_dict_new[node_type]
                h = torch.relu(h)
                h = self.norms[f'{node_type}_{layer_idx}'](h)
                h = self.dropout(h)
                
                # Residual connection (if dimensions match)
                if layer_idx > 0:
                    h = h + h_dict[node_type]
                
                h_dict_new[node_type] = h
            
            h_dict = h_dict_new
        
        # Final projection
        z_user = self.final_proj_user(h_dict['user'])
        z_place = self.final_proj_place(h_dict['place'])
        
        return {'user': z_user, 'place': z_place}

