#!/usr/bin/env python3
"""
Master script to generate all synthetic data.
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from recsys.synthetic.generator_config import get_default_config
from recsys.synthetic.generate_places import generate_all_places
from recsys.synthetic.generate_users import generate_all_users
from recsys.synthetic.generate_interactions import generate_all_interactions
from recsys.synthetic.generate_user_user_edges import generate_social_edges, generate_friend_labels


def save_to_parquet(data_list, output_path: Path, name: str):
    """Save list of dataclass objects to Parquet."""
    # Convert dataclass to dict
    data_dicts = []
    for item in data_list:
        d = {}
        for key, value in vars(item).items():
            if isinstance(value, datetime):
                d[key] = value.isoformat()
            else:
                d[key] = value
        data_dicts.append(d)
    
    df = pd.DataFrame(data_dicts)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for GNN training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=10_000,
        help="Number of users to generate"
    )
    parser.add_argument(
        "--n_places",
        type=int,
        default=10_000,
        help="Number of places to generate"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 80)
    
    # Load configuration
    config = get_default_config()
    config.N_USERS = args.n_users
    config.N_PLACES = args.n_places
    
    print(f"\nConfiguration:")
    print(f"  Users: {config.N_USERS}")
    print(f"  Places: {config.N_PLACES}")
    print(f"  Cities: {config.N_CITIES}")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print()
    
    # Step 1: Generate places
    print("Step 1/5: Generating places...")
    places = generate_all_places(config)
    save_to_parquet(places, output_dir / "places.parquet", "places")
    print()
    
    # Step 2: Generate users
    print("Step 2/5: Generating users...")
    users = generate_all_users(config)
    save_to_parquet(users, output_dir / "users.parquet", "users")
    print()
    
    # Step 3: Generate interactions
    print("Step 3/5: Generating interactions...")
    start_date = datetime(2024, 1, 1)
    interactions = generate_all_interactions(users, places, config, start_date)
    save_to_parquet(interactions, output_dir / "interactions.parquet", "interactions")
    print()
    
    # Step 4: Generate user-user edges
    print("Step 4/5: Generating social edges...")
    edges = generate_social_edges(users, interactions, config)
    save_to_parquet(edges, output_dir / "user_user_edges.parquet", "social edges")
    print()
    
    # Step 5: Generate friend labels
    print("Step 5/5: Generating friend labels...")
    friend_labels = generate_friend_labels(edges, users, config)
    save_to_parquet(friend_labels, output_dir / "friend_labels.parquet", "friend labels")
    print()
    
    # Summary
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated data:")
    print(f"  Users: {len(users)}")
    print(f"  Places: {len(places)}")
    print(f"  Interactions: {len(interactions)}")
    print(f"  Social edges: {len(edges)}")
    print(f"  Friend labels: {len(friend_labels)}")
    print(f"\nFiles saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Run scripts/run_build_features.py to build the graph")
    print("  2. Run scripts/run_train_gnn.py to train the model")
    print()


if __name__ == "__main__":
    main()

