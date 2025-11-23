#!/usr/bin/env python3
"""
Export trained embeddings to storage for serving.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import os
from recsys.features.graph_builder import load_graph
from recsys.ml.models.encoders import UserEncoder, PlaceEncoder
from recsys.ml.models.backbone import GraphRecBackbone
from recsys.config.model_config import ModelConfig


def export_embeddings(checkpoint_path: str, data_dir: str, output_dir: str):
    """
    Load trained model and export user/place embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config and graph
    config = ModelConfig()
    graph, user_id_to_index, place_id_to_index, index_to_user_id, index_to_place_id = load_graph(data_dir)
    
    # Load trained model
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    user_encoder = UserEncoder(config)
    place_encoder = PlaceEncoder(config)
    backbone = GraphRecBackbone(config)
    
    user_encoder.load_state_dict(checkpoint['user_encoder'])
    place_encoder.load_state_dict(checkpoint['place_encoder'])
    backbone.load_state_dict(checkpoint['backbone'])
    
    user_encoder.eval()
    place_encoder.eval()
    backbone.eval()
    
    # Compute embeddings
    print("Computing embeddings...")
    with torch.no_grad():
        # Encode features
        x_user = user_encoder(graph['user'].x)
        x_place = place_encoder(graph['place'].x)
        
        x_dict = {'user': x_user, 'place': x_place}
        
        # GNN forward pass
        z_dict = backbone(
            x_dict,
            graph.edge_index_dict,
            graph.edge_attr_dict
        )
        
        z_user = z_dict['user'].numpy()  # (N_users, D_MODEL)
        z_place = z_dict['place'].numpy()  # (N_places, D_MODEL)
    
    # Create dataframes with IDs
    print("Saving embeddings...")
    
    # User embeddings
    user_embeddings = []
    for idx in range(len(z_user)):
        user_id = index_to_user_id[idx]
        embedding = z_user[idx]
        user_embeddings.append({
            'user_id': user_id,
            'embedding': embedding.tolist()
        })
    
    user_df = pd.DataFrame(user_embeddings)
    user_df.to_parquet(f"{output_dir}/user_embeddings.parquet", index=False)
    
    # Place embeddings
    place_embeddings = []
    for idx in range(len(z_place)):
        place_id = index_to_place_id[idx]
        embedding = z_place[idx]
        place_embeddings.append({
            'place_id': place_id,
            'embedding': embedding.tolist()
        })
    
    place_df = pd.DataFrame(place_embeddings)
    place_df.to_parquet(f"{output_dir}/place_embeddings.parquet", index=False)
    
    print(f"✅ Exported {len(user_embeddings)} user embeddings")
    print(f"✅ Exported {len(place_embeddings)} place embeddings")
    print(f"Saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    export_embeddings(args.checkpoint, args.data_dir, args.output_dir)

