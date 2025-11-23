#!/usr/bin/env python3
"""
Main training script for GNN recommendation model.
"""

import torch
from torch.utils.data import DataLoader
from recsys.config.model_config import ModelConfig
from recsys.features.graph_builder import load_graph
from recsys.data.repositories import (
    UserRepository, PlaceRepository, InteractionRepository, FriendLabelRepository
)
from recsys.ml.models.encoders import UserEncoder, PlaceEncoder
from recsys.ml.models.backbone import GraphRecBackbone
from recsys.ml.models.heads import PlaceHead, FriendHead, ContextEncoder
from recsys.ml.training.datasets import PlaceRecommendationDataset, FriendCompatibilityDataset
from recsys.ml.training.train_loop import GNNTrainer
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = ModelConfig()
    
    # Load graph and data
    print("Loading graph...")
    graph, user_id_to_index, place_id_to_index, index_to_user_id, index_to_place_id = load_graph(args.data_dir)
    
    print("Loading data...")
    users = list(UserRepository(args.data_dir).get_all_users())
    places = list(PlaceRepository(args.data_dir).get_all_places())
    interactions = list(InteractionRepository(args.data_dir).get_all_interactions())
    friend_labels = list(FriendLabelRepository(args.data_dir).get_all_labels())
    
    # Create datasets
    print("Creating datasets...")
    place_dataset = PlaceRecommendationDataset(
        interactions, user_id_to_index, place_id_to_index, places
    )
    # Fill user_to_city mapping
    for user in users:
        place_dataset.user_to_city[user.user_id] = user.home_city_id
    
    friend_dataset = FriendCompatibilityDataset(friend_labels, user_id_to_index)
    
    # Data loaders
    place_loader = DataLoader(place_dataset, batch_size=config.BATCH_SIZE_PLACE, shuffle=True, num_workers=0)
    friend_loader = DataLoader(friend_dataset, batch_size=config.BATCH_SIZE_FRIEND, shuffle=True, num_workers=0)
    
    # Initialize models
    print("Initializing models...")
    user_encoder = UserEncoder(config)
    place_encoder = PlaceEncoder(config)
    backbone = GraphRecBackbone(config)
    place_head = PlaceHead(config)
    friend_head = FriendHead(config)
    place_ctx_encoder = ContextEncoder(config.D_CTX_PLACE)
    friend_ctx_encoder = ContextEncoder(config.D_CTX_FRIEND)
    
    # Initialize trainer
    trainer = GNNTrainer(
        user_encoder, place_encoder, backbone,
        place_head, friend_head,
        place_ctx_encoder, friend_ctx_encoder,
        graph, config, device=args.device
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        losses = trainer.train_epoch(place_loader, friend_loader, epoch)
        print(f"Epoch {epoch+1}/{args.epochs}: {losses}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    # Final save
    trainer.save_checkpoint(f"{args.output_dir}/final_model.pt")
    print("Training complete!")
    print(f"Model saved to {args.output_dir}/final_model.pt")


if __name__ == '__main__':
    main()

