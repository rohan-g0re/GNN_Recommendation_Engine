import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
import numpy as np
from recsys.data.schemas import UserSchema, PlaceSchema, InteractionSchema, UserUserEdgeSchema
from recsys.config.constants import (
    N_TIME_OF_DAY, N_DAY_OF_WEEK, MAX_NEIGHBORHOODS_PER_USER
)


def dict_to_fixed_vector(sparse_dict: Dict[int, float], size: int) -> List[float]:
    """
    Convert sparse neighborhood distribution to fixed-size vector.
    Takes top-k neighborhoods by weight, pads with zeros.
    """
    sorted_items = sorted(sparse_dict.items(), key=lambda x: -x[1])
    top_k = sorted_items[:size]
    
    vec = [0.0] * size
    for idx, (_, weight) in enumerate(top_k):
        vec[idx] = weight
    
    return vec


def encode_user_features(user: UserSchema, config) -> np.ndarray:
    """
    Convert UserSchema to feature vector for GNN input.
    
    Returns:
        np.ndarray of shape (D_user_raw,) where:
        D_user_raw = 2 (location IDs) + C_COARSE + C_FINE + C_VIBE + 
                     MAX_NEIGHBORHOODS_PER_USER + 5 (behavior stats)
                   = 2 + 6 + 100 + 30 + 5 + 5 = 148
    """
    features = []
    
    # Categorical location IDs (will be embedded in encoder)
    features.extend([
        float(user.home_city_id),
        float(user.home_neighborhood_id)
    ])
    
    # Preference vectors (already normalized)
    features.extend(user.cat_pref)        # 6 dims
    features.extend(user.fine_pref)       # 100 dims
    features.extend(user.vibe_pref)       # 30 dims
    
    # Area frequencies (convert dict to fixed-size vector)
    area_vec = dict_to_fixed_vector(
        user.area_freqs, 
        size=MAX_NEIGHBORHOODS_PER_USER  # 5
    )
    features.extend(area_vec)
    
    # Behavioral statistics (normalize to [0, 1] ranges)
    features.extend([
        min(user.avg_sessions_per_week / 10.0, 1.0),
        min(user.avg_views_per_session / 100.0, 1.0),
        min(user.avg_likes_per_session / 10.0, 1.0),
        min(user.avg_saves_per_session / 10.0, 1.0),
        min(user.avg_attends_per_month / 20.0, 1.0)
    ])
    
    return np.array(features, dtype=np.float32)


def encode_place_features(place: PlaceSchema, config) -> np.ndarray:
    """
    Convert PlaceSchema to feature vector.
    
    Returns:
        np.ndarray of shape (D_place_raw,) where:
        D_place_raw = 2 (location) + C_COARSE + C_FINE + 2 (price, time_slot) + 4 (popularity)
                    = 2 + 6 + 100 + 2 + 4 = 114
    """
    features = []
    
    # Location IDs (will be embedded)
    features.extend([
        float(place.city_id),
        float(place.neighborhood_id)
    ])
    
    # Category encoding
    features.extend(place.category_one_hot)  # 6 dims
    
    # Fine-grained tags
    features.extend(place.fine_tag_vector)   # 100 dims
    
    # Operational attributes (categorical, will be embedded)
    features.append(float(place.price_band))
    features.append(float(place.typical_time_slot))
    
    # Popularity metrics (log-normalize)
    features.extend([
        np.log1p(place.base_popularity) / 5.0,  # Normalize assuming max ~150
        np.log1p(place.avg_daily_visits) / 5.0,
        place.conversion_rate,  # Already in [0, 1]
        place.novelty_score     # Already in [0, 1]
    ])
    
    return np.array(features, dtype=np.float32)


def encode_interaction_edge_features(interaction: InteractionSchema) -> np.ndarray:
    """
    Convert interaction to edge feature vector.
    
    Returns:
        np.ndarray of shape (D_edge_up,) where:
        D_edge_up = 1 (rating) + 1 (dwell) + 4 (actions) + 4 (time_of_day_onehot) + 2 (dow_onehot)
                  = 12
    """
    features = []
    
    # Implicit rating (already 1-5)
    features.append(interaction.implicit_rating / 5.0)  # Normalize to [0, 1]
    
    # Dwell time (log normalize)
    features.append(min(np.log1p(interaction.dwell_time) / 6.0, 1.0))
    
    # Action counts (clip and normalize)
    features.extend([
        min(interaction.num_likes / 5.0, 1.0),
        min(interaction.num_saves / 3.0, 1.0),
        min(interaction.num_shares / 2.0, 1.0),
        float(interaction.attended)
    ])
    
    # Time of day (one-hot, 4 buckets)
    tod_onehot = [0.0] * N_TIME_OF_DAY
    if 0 <= interaction.time_of_day_bucket < N_TIME_OF_DAY:
        tod_onehot[interaction.time_of_day_bucket] = 1.0
    features.extend(tod_onehot)
    
    # Day of week (one-hot, 2 buckets)
    dow_onehot = [0.0] * N_DAY_OF_WEEK
    if 0 <= interaction.day_of_week_bucket < N_DAY_OF_WEEK:
        dow_onehot[interaction.day_of_week_bucket] = 1.0
    features.extend(dow_onehot)
    
    return np.array(features, dtype=np.float32)


def encode_social_edge_features(edge: UserUserEdgeSchema) -> np.ndarray:
    """
    Convert user-user edge to feature vector.
    
    Returns:
        np.ndarray of shape (D_edge_uu,) = 3
    """
    return np.array([
        edge.interest_overlap_score,  # [0, 1]
        min(edge.co_attendance_count / 10.0, 1.0),  # Normalize
        edge.same_neighborhood_freq   # [0, 1]
    ], dtype=np.float32)


def build_hetero_graph(
    users: List[UserSchema],
    places: List[PlaceSchema],
    interactions: List[InteractionSchema],
    user_user_edges: List[UserUserEdgeSchema],
    config
) -> Tuple[HeteroData, Dict, Dict, Dict, Dict]:
    """
    Build PyTorch Geometric HeteroData object.
    
    Returns:
        - HeteroData: graph object
        - user_id_to_index: mapping dict
        - place_id_to_index: mapping dict
        - index_to_user_id: reverse mapping
        - index_to_place_id: reverse mapping
    """
    
    # Step 1: Build ID mappings
    user_id_to_index = {user.user_id: idx for idx, user in enumerate(users)}
    index_to_user_id = {idx: user.user_id for idx, user in enumerate(users)}
    
    place_id_to_index = {place.place_id: idx for idx, place in enumerate(places)}
    index_to_place_id = {idx: place.place_id for idx, place in enumerate(places)}
    
    # Step 2: Build node features
    user_features = np.array([
        encode_user_features(user, config) for user in users
    ])  # Shape: (N_users, D_user_raw)
    
    place_features = np.array([
        encode_place_features(place, config) for place in places
    ])  # Shape: (N_places, D_place_raw)
    
    # Step 3: Build user-place edges
    up_edge_list = []
    up_edge_features = []
    
    for interaction in interactions:
        u_idx = user_id_to_index[interaction.user_id]
        p_idx = place_id_to_index[interaction.place_id]
        
        up_edge_list.append([u_idx, p_idx])
        up_edge_features.append(
            encode_interaction_edge_features(interaction)
        )
    
    up_edge_index = torch.tensor(up_edge_list, dtype=torch.long).t()  # Shape: (2, E_up)
    up_edge_attr = torch.tensor(np.array(up_edge_features), dtype=torch.float)  # (E_up, 12)
    
    # Reverse edges (place -> user)
    pu_edge_index = up_edge_index.flip(0)  # Flip source and target
    pu_edge_attr = up_edge_attr.clone()    # Same features
    
    # Step 4: Build user-user edges
    uu_edge_list = []
    uu_edge_features = []
    
    for edge in user_user_edges:
        u_idx = user_id_to_index[edge.user_u]
        v_idx = user_id_to_index[edge.user_v]
        
        # Add both directions for undirected edge
        uu_edge_list.append([u_idx, v_idx])
        uu_edge_list.append([v_idx, u_idx])
        
        feat = encode_social_edge_features(edge)
        uu_edge_features.append(feat)
        uu_edge_features.append(feat)  # Same features both directions
    
    uu_edge_index = torch.tensor(uu_edge_list, dtype=torch.long).t()  # (2, E_uu*2)
    uu_edge_attr = torch.tensor(np.array(uu_edge_features), dtype=torch.float)  # (E_uu*2, 3)
    
    # Step 5: Construct HeteroData
    data = HeteroData()
    
    # Node features
    data['user'].x = torch.tensor(user_features, dtype=torch.float)
    data['place'].x = torch.tensor(place_features, dtype=torch.float)
    
    # User-place edges
    data['user', 'interacts', 'place'].edge_index = up_edge_index
    data['user', 'interacts', 'place'].edge_attr = up_edge_attr
    
    # Place-user edges (reverse)
    data['place', 'rev_interacts', 'user'].edge_index = pu_edge_index
    data['place', 'rev_interacts', 'user'].edge_attr = pu_edge_attr
    
    # User-user edges
    data['user', 'social', 'user'].edge_index = uu_edge_index
    data['user', 'social', 'user'].edge_attr = uu_edge_attr
    
    # Store metadata
    data['user'].num_nodes = len(users)
    data['place'].num_nodes = len(places)
    
    return data, user_id_to_index, place_id_to_index, index_to_user_id, index_to_place_id


def save_graph(
    data: HeteroData,
    user_id_to_index: Dict,
    place_id_to_index: Dict,
    index_to_user_id: Dict,
    index_to_place_id: Dict,
    output_dir: str
):
    """
    Serialize graph and mappings to disk.
    """
    import pickle
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph
    torch.save(data, f"{output_dir}/hetero_graph.pt")
    
    # Save mappings
    with open(f"{output_dir}/user_id_mappings.pkl", 'wb') as f:
        pickle.dump({
            'id_to_index': user_id_to_index,
            'index_to_id': index_to_user_id
        }, f)
    
    with open(f"{output_dir}/place_id_mappings.pkl", 'wb') as f:
        pickle.dump({
            'id_to_index': place_id_to_index,
            'index_to_id': index_to_place_id
        }, f)


def load_graph(input_dir: str) -> Tuple[HeteroData, Dict, Dict, Dict, Dict]:
    """
    Load graph and mappings from disk.
    """
    import pickle
    
    data = torch.load(f"{input_dir}/hetero_graph.pt")
    
    with open(f"{input_dir}/user_id_mappings.pkl", 'rb') as f:
        user_maps = pickle.load(f)
    
    with open(f"{input_dir}/place_id_mappings.pkl", 'rb') as f:
        place_maps = pickle.load(f)
    
    return (
        data,
        user_maps['id_to_index'],
        place_maps['id_to_index'],
        user_maps['index_to_id'],
        place_maps['index_to_id']
    )

