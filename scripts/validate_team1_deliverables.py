#!/usr/bin/env python3
"""
Validate that Team 1 delivered all required files correctly.
Run this after receiving Team 1's deliverables.
"""

import pandas as pd
import torch
import pickle
from pathlib import Path
import sys


def validate_team1_deliverables():
    """Validate that Team 1 delivered everything correctly."""
    
    errors = []
    warnings = []
    
    # 1. Check files exist
    print("Checking required files...")
    required_files = [
        'data/embeddings/user_embeddings.parquet',
        'data/embeddings/place_embeddings.parquet',
        'models/final_model.pt'
    ]
    
    optional_files = [
        'data/user_id_mappings.pkl',
        'data/place_id_mappings.pkl',
        'data/users.parquet',
        'data/places.parquet',
    ]
    
    for file in required_files:
        if not Path(file).exists():
            errors.append(f"Missing required file: {file}")
        else:
            print(f"  ✅ {file}")
    
    for file in optional_files:
        if not Path(file).exists():
            warnings.append(f"Optional file missing: {file}")
        else:
            print(f"  ✅ {file} (optional)")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    # 2. Check embeddings
    print("\nValidating embeddings...")
    try:
        user_df = pd.read_parquet('data/embeddings/user_embeddings.parquet')
        assert 'user_id' in user_df.columns, "Missing 'user_id' column"
        assert 'embedding' in user_df.columns, "Missing 'embedding' column"
        
        if len(user_df) == 0:
            errors.append("User embeddings file is empty")
        else:
            embedding_dim = len(user_df.iloc[0]['embedding'])
            assert embedding_dim == 128, f"Wrong embedding dimension: {embedding_dim}, expected 128"
            print(f"  ✅ User embeddings: {len(user_df)} users, dimension {embedding_dim}")
    except Exception as e:
        errors.append(f"Error validating user embeddings: {e}")
    
    try:
        place_df = pd.read_parquet('data/embeddings/place_embeddings.parquet')
        assert 'place_id' in place_df.columns, "Missing 'place_id' column"
        assert 'embedding' in place_df.columns, "Missing 'embedding' column"
        
        if len(place_df) == 0:
            errors.append("Place embeddings file is empty")
        else:
            embedding_dim = len(place_df.iloc[0]['embedding'])
            assert embedding_dim == 128, f"Wrong embedding dimension: {embedding_dim}, expected 128"
            print(f"  ✅ Place embeddings: {len(place_df)} places, dimension {embedding_dim}")
    except Exception as e:
        errors.append(f"Error validating place embeddings: {e}")
    
    # 3. Check checkpoint
    print("\nValidating model checkpoint...")
    try:
        checkpoint = torch.load('models/final_model.pt', map_location='cpu')
        required_keys = ['place_head', 'friend_head', 'place_ctx_encoder', 'friend_ctx_encoder']
        
        for key in required_keys:
            if key not in checkpoint:
                errors.append(f"Missing key in checkpoint: {key}")
            else:
                print(f"  ✅ Checkpoint contains '{key}'")
    except Exception as e:
        errors.append(f"Error loading checkpoint: {e}")
    
    # 4. Check mappings (optional)
    if Path('data/user_id_mappings.pkl').exists():
        print("\nValidating ID mappings...")
        try:
            with open('data/user_id_mappings.pkl', 'rb') as f:
                user_maps = pickle.load(f)
            assert 'id_to_index' in user_maps, "Missing 'id_to_index' in user mappings"
            assert 'index_to_id' in user_maps, "Missing 'index_to_id' in user mappings"
            print(f"  ✅ User mappings: {len(user_maps['id_to_index'])} entries")
        except Exception as e:
            warnings.append(f"Error validating user mappings: {e}")
    
    if Path('data/place_id_mappings.pkl').exists():
        try:
            with open('data/place_id_mappings.pkl', 'rb') as f:
                place_maps = pickle.load(f)
            assert 'id_to_index' in place_maps, "Missing 'id_to_index' in place mappings"
            assert 'index_to_id' in place_maps, "Missing 'index_to_id' in place mappings"
            print(f"  ✅ Place mappings: {len(place_maps['id_to_index'])} entries")
        except Exception as e:
            warnings.append(f"Error validating place mappings: {e}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print("❌ VALIDATION FAILED")
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ VALIDATION PASSED")
        if warnings:
            print("\nWarnings (non-critical):")
            for warning in warnings:
                print(f"  - {warning}")
        print("\n🎉 All deliverables validated! Ready for integration.")
        return True


if __name__ == "__main__":
    success = validate_team1_deliverables()
    sys.exit(0 if success else 1)

