#!/usr/bin/env python3
"""
Build graph from synthetic data.
"""

import argparse
from pathlib import Path
from recsys.features.graph_builder import build_hetero_graph, save_graph
from recsys.data.repositories import (
    UserRepository, PlaceRepository, InteractionRepository, UserUserEdgeRepository
)
from recsys.config.constants import MAX_NEIGHBORHOODS_PER_USER


class Config:
    """Simple config for graph building."""
    MAX_NEIGHBORHOODS_PER_USER = MAX_NEIGHBORHOODS_PER_USER


def main():
    parser = argparse.ArgumentParser(description="Build graph from synthetic data")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with parquet files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for graph')
    args = parser.parse_args()
    
    print("Loading data...")
    users = list(UserRepository(args.data_dir).get_all_users())
    places = list(PlaceRepository(args.data_dir).get_all_places())
    interactions = list(InteractionRepository(args.data_dir).get_all_interactions())
    user_user_edges = list(UserUserEdgeRepository(args.data_dir).get_all_edges())
    
    print(f"Loaded:")
    print(f"  Users: {len(users)}")
    print(f"  Places: {len(places)}")
    print(f"  Interactions: {len(interactions)}")
    print(f"  User-user edges: {len(user_user_edges)}")
    
    print("\nBuilding graph...")
    config = Config()
    graph, user_id_to_index, place_id_to_index, index_to_user_id, index_to_place_id = build_hetero_graph(
        users, places, interactions, user_user_edges, config
    )
    
    print(f"Graph built:")
    print(f"  User nodes: {graph['user'].x.shape}")
    print(f"  Place nodes: {graph['place'].x.shape}")
    print(f"  User-place edges: {graph['user', 'interacts', 'place'].edge_index.shape[1]}")
    print(f"  User-user edges: {graph['user', 'social', 'user'].edge_index.shape[1]}")
    
    print("\nSaving graph...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_graph(
        graph, user_id_to_index, place_id_to_index,
        index_to_user_id, index_to_place_id,
        args.output_dir
    )
    
    print(f"✅ Graph saved to {args.output_dir}")


if __name__ == '__main__':
    main()

