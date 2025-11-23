#!/usr/bin/env python3
"""
Verification script to check that all deliverables exist and are correct.
"""

import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import torch


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    if exists:
        print(f"✅ {description}: {filepath}")
    else:
        print(f"❌ MISSING: {description}: {filepath}")
    return exists


def verify_parquet_schema(filepath: str, expected_columns: list, description: str) -> bool:
    """Verify parquet file has correct schema."""
    try:
        df = pd.read_parquet(filepath)
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            print(f"❌ {description}: Missing columns {missing_cols}")
            return False
        
        print(f"✅ {description}: Schema correct ({len(df)} rows)")
        return True
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False


def verify_embeddings(filepath: str, id_col: str, description: str) -> bool:
    """Verify embeddings file has correct format."""
    try:
        df = pd.read_parquet(filepath)
        
        # Check columns
        if id_col not in df.columns or 'embedding' not in df.columns:
            print(f"❌ {description}: Missing required columns")
            return False
        
        # Check embedding dimensions
        sample_embedding = df.iloc[0]['embedding']
        if not isinstance(sample_embedding, list):
            print(f"❌ {description}: Embedding is not a list")
            return False
        
        if len(sample_embedding) != 128:
            print(f"❌ {description}: Embedding dimension is {len(sample_embedding)}, expected 128")
            return False
        
        print(f"✅ {description}: {len(df)} embeddings, dimension={len(sample_embedding)}")
        return True
    except Exception as e:
        print(f"❌ {description}: Error - {e}")
        return False


def verify_checkpoint(filepath: str) -> bool:
    """Verify model checkpoint has all required components."""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        required_keys = [
            'user_encoder',
            'place_encoder',
            'backbone',
            'place_head',
            'friend_head',
            'place_ctx_encoder',
            'friend_ctx_encoder',
            'config'
        ]
        
        missing_keys = set(required_keys) - set(checkpoint.keys())
        if missing_keys:
            print(f"❌ Checkpoint: Missing keys {missing_keys}")
            return False
        
        print(f"✅ Checkpoint: All required components present")
        return True
    except Exception as e:
        print(f"❌ Checkpoint: Error loading - {e}")
        return False


def verify_mappings(filepath: str, description: str) -> bool:
    """Verify ID mappings pickle file."""
    try:
        with open(filepath, 'rb') as f:
            mappings = pickle.load(f)
        
        if 'id_to_index' not in mappings or 'index_to_id' not in mappings:
            print(f"❌ {description}: Missing required keys")
            return False
        
        print(f"✅ {description}: {len(mappings['id_to_index'])} mappings")
        return True
    except Exception as e:
        print(f"❌ {description}: Error loading - {e}")
        return False


def main():
    print("=" * 80)
    print("VERIFYING GNN TRAINING PIPELINE DELIVERABLES")
    print("=" * 80)
    print()
    
    # Check if data directory exists
    data_dir = Path("data")
    models_dir = Path("models")
    embeddings_dir = data_dir / "embeddings"
    
    if not data_dir.exists():
        print("❌ Data directory does not exist. Run synthetic data generation first.")
        return False
    
    all_ok = True
    
    # 1. Check synthetic data files
    print("Checking synthetic data files...")
    all_ok &= check_file_exists(data_dir / "users.parquet", "Users metadata")
    all_ok &= check_file_exists(data_dir / "places.parquet", "Places metadata")
    all_ok &= check_file_exists(data_dir / "interactions.parquet", "Interactions")
    all_ok &= check_file_exists(data_dir / "user_user_edges.parquet", "User-user edges")
    all_ok &= check_file_exists(data_dir / "friend_labels.parquet", "Friend labels")
    print()
    
    # 2. Check graph files
    print("Checking graph files...")
    all_ok &= check_file_exists(data_dir / "hetero_graph.pt", "Graph file")
    all_ok &= check_file_exists(data_dir / "user_id_mappings.pkl", "User ID mappings")
    all_ok &= check_file_exists(data_dir / "place_id_mappings.pkl", "Place ID mappings")
    print()
    
    # 3. Check model checkpoint
    print("Checking model checkpoint...")
    if check_file_exists(models_dir / "final_model.pt", "Model checkpoint"):
        all_ok &= verify_checkpoint(models_dir / "final_model.pt")
    else:
        all_ok = False
    print()
    
    # 4. Check embeddings
    print("Checking embeddings...")
    if embeddings_dir.exists():
        all_ok &= verify_embeddings(
            embeddings_dir / "user_embeddings.parquet",
            "user_id",
            "User embeddings"
        )
        all_ok &= verify_embeddings(
            embeddings_dir / "place_embeddings.parquet",
            "place_id",
            "Place embeddings"
        )
    else:
        print("❌ Embeddings directory does not exist")
        all_ok = False
    print()
    
    # 5. Verify schemas
    print("Verifying file schemas...")
    if (data_dir / "users.parquet").exists():
        verify_parquet_schema(
            data_dir / "users.parquet",
            ['user_id', 'home_city_id', 'home_neighborhood_id', 'cat_pref', 
             'fine_pref', 'vibe_pref', 'area_freqs', 'avg_sessions_per_week',
             'avg_views_per_session', 'avg_likes_per_session', 'avg_saves_per_session',
             'avg_attends_per_month'],
            "Users schema"
        )
    
    if (data_dir / "places.parquet").exists():
        verify_parquet_schema(
            data_dir / "places.parquet",
            ['place_id', 'city_id', 'neighborhood_id', 'category_ids',
             'category_one_hot', 'fine_tag_vector', 'price_band', 'typical_time_slot',
             'base_popularity', 'avg_daily_visits', 'conversion_rate', 'novelty_score'],
            "Places schema"
        )
    print()
    
    # Summary
    print("=" * 80)
    if all_ok:
        print("✅ ALL DELIVERABLES VERIFIED")
        print()
        print("Deliverables checklist:")
        print("  ✅ data/embeddings/user_embeddings.parquet")
        print("  ✅ data/embeddings/place_embeddings.parquet")
        print("  ✅ models/final_model.pt")
        print("  ✅ data/user_id_mappings.pkl")
        print("  ✅ data/place_id_mappings.pkl")
        print("  ✅ data/users.parquet")
        print("  ✅ data/places.parquet")
        return True
    else:
        print("❌ SOME DELIVERABLES MISSING OR INCORRECT")
        print()
        print("Run the pipeline:")
        print("  1. python recsys/scripts/run_synthetic_generation.py --output_dir data/")
        print("  2. python recsys/scripts/run_build_features.py --data_dir data/ --output_dir data/")
        print("  3. python recsys/scripts/run_train_gnn.py --data_dir data/ --output_dir models/ --epochs 50")
        print("  4. python recsys/scripts/run_export_embeddings.py --checkpoint models/final_model.pt --data_dir data/ --output_dir data/embeddings/")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

