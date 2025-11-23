#!/usr/bin/env python3
"""
Build ANN indices from embeddings for fast retrieval.
Groups embeddings by city and builds separate indices per city.
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

from recsys.serving.ann_index import CityAnnIndexManager
from recsys.config.model_config import ModelConfig
from recsys.config.constants import N_CITIES

# Try to import repositories, create stubs if they don't exist
try:
    from recsys.data.repositories import UserRepository, PlaceRepository
except ImportError:
    # Create minimal stub repositories if they don't exist yet
    class UserRepository:
        def __init__(self, data_dir: str):
            self.data_dir = data_dir
        
        def get_user(self, user_id: int):
            # Stub - should be implemented by training team
            return None
    
    class PlaceRepository:
        def __init__(self, data_dir: str):
            self.data_dir = data_dir
        
        def get_place(self, place_id: int):
            # Stub - should be implemented by training team
            return None


def build_indices(
    embeddings_dir: str,
    output_dir: str,
    data_dir: str = "data"
):
    """
    Build ANN indices from embeddings, grouped by city.
    
    Args:
        embeddings_dir: Directory containing user_embeddings.parquet and place_embeddings.parquet
        output_dir: Directory to save indices
        data_dir: Directory containing user/place metadata (for city mapping)
    """
    config = ModelConfig()
    
    print("Loading embeddings...")
    # Load embeddings
    user_df = pd.read_parquet(f"{embeddings_dir}/user_embeddings.parquet")
    place_df = pd.read_parquet(f"{embeddings_dir}/place_embeddings.parquet")
    
    print(f"Loaded {len(user_df)} user embeddings")
    print(f"Loaded {len(place_df)} place embeddings")
    
    # Load repositories for city mapping
    try:
        user_repo = UserRepository(data_dir)
        place_repo = PlaceRepository(data_dir)
    except Exception as e:
        print(f"Warning: Could not load repositories: {e}")
        print("Will attempt to infer city from embeddings or use default city=0")
        user_repo = None
        place_repo = None
    
    # Group embeddings by city
    print("Grouping embeddings by city...")
    
    # For users: group by home_city_id
    user_embeddings_by_city = defaultdict(lambda: {'embeddings': [], 'ids': []})
    for _, row in user_df.iterrows():
        user_id = row['user_id']
        embedding = np.array(row['embedding'])
        
        # Try to get city from repository
        city_id = 0  # Default
        if user_repo is not None:
            try:
                user = user_repo.get_user(user_id)
                if user is not None:
                    city_id = user.home_city_id
            except:
                pass
        
        user_embeddings_by_city[city_id]['embeddings'].append(embedding)
        user_embeddings_by_city[city_id]['ids'].append(user_id)
    
    # For places: group by city_id
    place_embeddings_by_city = defaultdict(lambda: {'embeddings': [], 'ids': []})
    for _, row in place_df.iterrows():
        place_id = row['place_id']
        embedding = np.array(row['embedding'])
        
        # Try to get city from repository
        city_id = 0  # Default
        if place_repo is not None:
            try:
                place = place_repo.get_place(place_id)
                if place is not None:
                    city_id = place.city_id
            except:
                pass
        
        place_embeddings_by_city[city_id]['embeddings'].append(embedding)
        place_embeddings_by_city[city_id]['ids'].append(place_id)
    
    # Build indices
    print("Building ANN indices...")
    
    # User indices
    user_ann_manager = CityAnnIndexManager(dimension=config.D_MODEL)
    for city_id in range(N_CITIES):
        if city_id in user_embeddings_by_city:
            data = user_embeddings_by_city[city_id]
            embeddings_array = np.array(data['embeddings'])
            ids_list = data['ids']
            
            print(f"Building user index for city {city_id} ({len(ids_list)} users)...")
            user_ann_manager.build_city_index(city_id, embeddings_array, ids_list)
        else:
            print(f"No users found for city {city_id}, skipping...")
    
    # Place indices
    place_ann_manager = CityAnnIndexManager(dimension=config.D_MODEL)
    for city_id in range(N_CITIES):
        if city_id in place_embeddings_by_city:
            data = place_embeddings_by_city[city_id]
            embeddings_array = np.array(data['embeddings'])
            ids_list = data['ids']
            
            print(f"Building place index for city {city_id} ({len(ids_list)} places)...")
            place_ann_manager.build_city_index(city_id, embeddings_array, ids_list)
        else:
            print(f"No places found for city {city_id}, skipping...")
    
    # Save indices
    print(f"Saving indices to {output_dir}...")
    user_ann_manager.save(output_dir, "user")
    place_ann_manager.save(output_dir, "place")
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build ANN indices from embeddings')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing user_embeddings.parquet and place_embeddings.parquet')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save indices')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing user/place metadata (for city mapping)')
    
    args = parser.parse_args()
    
    build_indices(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )

