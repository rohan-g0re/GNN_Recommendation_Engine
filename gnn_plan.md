## 1. Overview and Goals

This document defines a **concrete end-to-end plan for the Graph Neural Network (GNN)–based recommendation engine**, including:

- **Data and signals** we will model.
- The **graph schema** (nodes, edges, features).
- The **GNN architecture** (backbone + task heads).
- **Training objectives** and losses.
- **Inference and serving** strategy.
- A detailed **synthetic data generation plan** aligned with the above.

The goal is to directly build a **GraphRec-style heterogeneous GNN** that powers:

- **Place recommendations** (spatial analysis).
- **People recommendations** (social compatibility + incentive to attend).

We assume on the order of **10k users** and **10k places**, across several cities, and we design for easy scaling upwards.

---

## 2. Data Models and Feature Specifications

### 2.1 Global Constants and Taxonomy

**CRITICAL: These constants MUST match between GNN training and serving layers.**

```python
# File: recsys/config/constants.py

# Scale parameters
N_USERS = 10_000
N_PLACES = 10_000
N_CITIES = 8
N_NEIGHBORHOODS_PER_CITY = 15

# Feature dimensions
C_COARSE = 6  # Number of coarse categories
C_FINE = 100  # Number of fine-grained tags
C_VIBE = 30   # Number of vibe/personality tags

# Coarse categories (index-based, 0-indexed)
COARSE_CATEGORIES = [
    "entertainment",  # 0
    "sports",         # 1
    "clubs",          # 2
    "dining",         # 3
    "outdoors",       # 4
    "culture"         # 5
]

# Fine tags (0-indexed, first 20 shown as example)
FINE_TAGS = [
    "fishing", "bouldering", "techno", "live_music", "board_games",
    "rooftop", "brunch", "karaoke", "jazz", "trivia",
    "wine_tasting", "comedy", "art_gallery", "theater", "dance",
    "pottery", "yoga", "crossfit", "rock_climbing", "cycling",
    # ... 80 more tags to total 100
]

# Vibe tags (0-indexed, first 15 shown)
VIBE_TAGS = [
    "introvert", "extrovert", "party", "fitness", "artsy",
    "night_owl", "early_bird", "foodie", "adventurous", "chill",
    "intellectual", "spontaneous", "planner", "social", "independent",
    # ... 15 more to total 30
]

# Time slot buckets
TIME_SLOTS = ["morning", "afternoon", "evening", "night", "weekend_day", "weekend_night"]
N_TIME_SLOTS = len(TIME_SLOTS)

# Time of day buckets (for interactions)
TIME_OF_DAY_BUCKETS = ["morning", "afternoon", "evening", "night"]
N_TIME_OF_DAY = 4

# Day of week buckets
DAY_OF_WEEK_BUCKETS = ["weekday", "weekend"]
N_DAY_OF_WEEK = 2

# Price bands
N_PRICE_BANDS = 5  # 0-4

# Neighborhood representation
MAX_NEIGHBORHOODS_PER_USER = 5  # Fixed-size vector for area_freqs
```

### 2.2 User Data Model (Raw Schema)

**Storage format**: `users.parquet` or PostgreSQL table `users`

**Python schema** (in `recsys/data/schemas.py`):

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class UserSchema:
    """
    Raw user data schema.
    Maps 1:1 to storage (CSV/Parquet/DB).
    """
    # Identity
    user_id: int  # Primary key, 0 to N_USERS-1
    
    # Location attributes (categorical IDs)
    home_city_id: int  # 0 to N_CITIES-1
    home_neighborhood_id: int  # 0 to N_NEIGHBORHOODS_PER_CITY-1 within city
    
    # Preference vectors (pre-computed, normalized)
    cat_pref: List[float]  # Length C_COARSE=6, sums to 1.0
    fine_pref: List[float]  # Length C_FINE=100, sums to 1.0
    vibe_pref: List[float]  # Length C_VIBE=30, sums to 1.0
    
    # Location behavior (sparse or dense)
    # Option 1: Dict mapping neighborhood_id -> weight
    area_freqs: Dict[int, float]  # Keys: neighborhood_ids, values sum to 1.0
    # Will be converted to fixed-size vector in feature engineering
    
    # Behavioral statistics (continuous)
    avg_sessions_per_week: float  # e.g., 2.5
    avg_views_per_session: float  # e.g., 25.0
    avg_likes_per_session: float  # e.g., 3.0
    avg_saves_per_session: float  # e.g., 1.0
    avg_attends_per_month: float  # e.g., 4.0
    
    # Optional demographics (for future extension)
    age_group: Optional[int] = None  # e.g., 0-5 for age buckets
    gender: Optional[int] = None  # e.g., 0/1/2
```

**Storage columns** (Parquet):
- `user_id`: int64
- `home_city_id`: int32
- `home_neighborhood_id`: int32
- `cat_pref`: list<float> (length 6)
- `fine_pref`: list<float> (length 100)
- `vibe_pref`: list<float> (length 30)
- `area_freqs`: map<int32, float>
- `avg_sessions_per_week`: float32
- `avg_views_per_session`: float32
- `avg_likes_per_session`: float32
- `avg_saves_per_session`: float32
- `avg_attends_per_month`: float32

### 2.3 Place Data Model (Raw Schema)

**Storage format**: `places.parquet` or table `places`

**Python schema**:

```python
@dataclass
class PlaceSchema:
    """
    Raw place data schema.
    """
    # Identity
    place_id: int  # Primary key, 0 to N_PLACES-1
    
    # Location
    city_id: int  # 0 to N_CITIES-1
    neighborhood_id: int  # 0 to N_NEIGHBORHOODS_PER_CITY-1
    
    # Category (can be multi-label)
    category_ids: List[int]  # List of category indices, e.g., [0, 2] for entertainment+clubs
    category_one_hot: List[float]  # Length C_COARSE=6, binary or weighted
    
    # Fine-grained tags
    fine_tag_vector: List[float]  # Length C_FINE=100, normalized weights
    
    # Operational attributes
    price_band: int  # 0-4
    typical_time_slot: int  # Index into TIME_SLOTS (0-5)
    
    # Popularity metrics
    base_popularity: float  # Sampled from log-normal, e.g., 0.1 to 10.0
    avg_daily_visits: float  # Derived from base_popularity
    conversion_rate: float  # Fraction of views that lead to attendance
    novelty_score: float  # 1.0 / (1 + log(base_popularity))
```

**Storage columns**:
- `place_id`: int64
- `city_id`: int32
- `neighborhood_id`: int32
- `category_ids`: list<int32>
- `category_one_hot`: list<float> (length 6)
- `fine_tag_vector`: list<float> (length 100)
- `price_band`: int32
- `typical_time_slot`: int32
- `base_popularity`: float32
- `avg_daily_visits`: float32
- `conversion_rate`: float32
- `novelty_score`: float32

### 2.4 Interaction Data Model (User–Place Edge Data)

**Storage format**: `interactions.parquet` or table `interactions`

**Python schema**:

```python
from datetime import datetime

@dataclass
class InteractionSchema:
    """
    User-place interaction event.
    Represents one view/engagement session.
    """
    # Foreign keys
    user_id: int  # References UserSchema.user_id
    place_id: int  # References PlaceSchema.place_id
    
    # Engagement metrics
    dwell_time: float  # Seconds spent viewing posts about this place
    num_likes: int
    num_saves: int
    num_shares: int
    attended: bool  # True if user attended event at this place
    
    # Derived implicit rating
    implicit_rating: float  # 1.0-5.0, derived from above metrics
    
    # Temporal context
    timestamp: datetime  # When interaction occurred
    time_of_day_bucket: int  # 0-3 (morning/afternoon/evening/night)
    day_of_week_bucket: int  # 0-1 (weekday/weekend)
```

**Storage columns**:
- `user_id`: int64
- `place_id`: int64
- `dwell_time`: float32
- `num_likes`: int32
- `num_saves`: int32
- `num_shares`: int32
- `attended`: bool
- `implicit_rating`: float32
- `timestamp`: timestamp
- `time_of_day_bucket`: int32
- `day_of_week_bucket`: int32

**Implicit rating derivation formula** (for synthetic generation and preprocessing):

```python
def compute_implicit_rating(
    dwell_time: float,
    num_likes: int,
    num_saves: int,
    num_shares: int,
    attended: bool
) -> float:
    """
    Maps engagement signals to 1-5 rating scale.
    
    Logic:
    - Base score from normalized dwell time (0-2 points)
    - +0.5 per like (up to 1.5)
    - +1.0 per save (up to 2.0)
    - +0.5 per share (up to 1.0)
    - +2.0 if attended
    
    Capped at 5.0, floored at 1.0
    """
    score = 1.0
    
    # Dwell contribution (normalize assuming max ~300 seconds = 5 min)
    score += min(dwell_time / 150.0, 2.0)
    
    # Action contributions
    score += min(num_likes * 0.5, 1.5)
    score += min(num_saves * 1.0, 2.0)
    score += min(num_shares * 0.5, 1.0)
    
    if attended:
        score += 2.0
    
    return min(score, 5.0)
```

### 2.5 User–User Edge Data Model (Social Edge Data)

**Storage format**: `user_user_edges.parquet` or table `user_user_edges`

**Python schema**:

```python
@dataclass
class UserUserEdgeSchema:
    """
    Soft social edge between two users.
    Stored as undirected (single row per pair, user_u < user_v).
    """
    user_u: int  # First user ID (smaller ID)
    user_v: int  # Second user ID (larger ID)
    
    # Edge strength features
    interest_overlap_score: float  # Cosine similarity of fine_pref, [0, 1]
    co_attendance_count: int  # Number of places both attended
    same_neighborhood_freq: float  # Jaccard or overlap of area_freqs, [0, 1]
```

**Storage columns**:
- `user_u`: int64
- `user_v`: int64
- `interest_overlap_score`: float32
- `co_attendance_count`: int32
- `same_neighborhood_freq`: float32

### 2.6 Friend Label Data Model (Supervision for Friend Head)

**Storage format**: `friend_labels.parquet` or table `friend_labels`

**Python schema**:

```python
@dataclass
class FriendLabelSchema:
    """
    Labeled user pairs for training friend compatibility head.
    """
    user_u: int
    user_v: int
    
    # Binary labels
    label_compat: int  # 0 (negative) or 1 (positive) for compatibility
    label_attend: int  # 0 (won't attend) or 1 (will attend) if invited together
```

**Label generation rules**:
- **Positive compatibility** (`label_compat=1`):
  - `interest_overlap_score >= 0.6` OR `co_attendance_count >= 2`
- **Negative compatibility** (`label_compat=0`):
  - `interest_overlap_score < 0.3` AND `co_attendance_count == 0`
- **Attendance label** (`label_attend`):
  - For positive pairs: sample from Bernoulli with probability proportional to `interest_overlap_score * (1 + co_attendance_count/10)`
  - For negative pairs: always 0

### 2.7 Data Contracts and Validation

**Validation rules** (implement in `recsys/data/validators.py`):

```python
def validate_user_schema(user: UserSchema) -> None:
    """Validate user data integrity."""
    assert 0 <= user.user_id < N_USERS
    assert 0 <= user.home_city_id < N_CITIES
    assert 0 <= user.home_neighborhood_id < N_NEIGHBORHOODS_PER_CITY
    
    assert len(user.cat_pref) == C_COARSE
    assert abs(sum(user.cat_pref) - 1.0) < 1e-5, "cat_pref must sum to 1"
    
    assert len(user.fine_pref) == C_FINE
    assert abs(sum(user.fine_pref) - 1.0) < 1e-5, "fine_pref must sum to 1"
    
    assert len(user.vibe_pref) == C_VIBE
    assert abs(sum(user.vibe_pref) - 1.0) < 1e-5, "vibe_pref must sum to 1"
    
    assert all(v >= 0 for v in user.area_freqs.values())
    assert abs(sum(user.area_freqs.values()) - 1.0) < 1e-5
    
    assert user.avg_sessions_per_week > 0
    assert user.avg_views_per_session > 0

def validate_place_schema(place: PlaceSchema) -> None:
    """Validate place data integrity."""
    assert 0 <= place.place_id < N_PLACES
    assert 0 <= place.city_id < N_CITIES
    assert 0 <= place.neighborhood_id < N_NEIGHBORHOODS_PER_CITY
    
    assert len(place.category_one_hot) == C_COARSE
    assert all(0 <= v <= 1 for v in place.category_one_hot)
    
    assert len(place.fine_tag_vector) == C_FINE
    assert abs(sum(place.fine_tag_vector) - 1.0) < 1e-5
    
    assert 0 <= place.price_band < N_PRICE_BANDS
    assert 0 <= place.typical_time_slot < N_TIME_SLOTS
```

---

## 3. PyTorch Geometric Graph Construction

### 3.1 Graph Structure Overview

We construct a **heterogeneous graph** using PyTorch Geometric's `HeteroData` object.

**Node types**: `['user', 'place']`

**Edge types**: 
- `('user', 'interacts', 'place')` – user views/engages with place
- `('place', 'rev_interacts', 'user')` – reverse of above for message passing
- `('user', 'social', 'user')` – soft social connections

### 3.2 Feature Tensor Specifications

**File**: `recsys/features/graph_builder.py`

#### 3.2.1 User Node Features

**Raw feature composition**:

```python
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
        size=config.MAX_NEIGHBORHOODS_PER_USER  # 5
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
```

**Tensor shape in graph**:
- `data['user'].x`: `(N_users, D_user_raw)` = `(10000, 148)`

#### 3.2.2 Place Node Features

```python
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
```

**Tensor shape**:
- `data['place'].x`: `(N_places, D_place_raw)` = `(10000, 114)`

#### 3.2.3 User–Place Edge Features

```python
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
```

**Tensor shape**:
- `data['user', 'interacts', 'place'].edge_attr`: `(E_up, 12)`
- `data['place', 'rev_interacts', 'user'].edge_attr`: `(E_up, 12)` (same features)

#### 3.2.4 User–User Edge Features

```python
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
```

**Tensor shape**:
- `data['user', 'social', 'user'].edge_attr`: `(E_uu, 3)`

### 3.3 Complete Graph Builder Implementation

**File**: `recsys/features/graph_builder.py`

```python
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
import numpy as np

def build_hetero_graph(
    users: List[UserSchema],
    places: List[PlaceSchema],
    interactions: List[InteractionSchema],
    user_user_edges: List[UserUserEdgeSchema],
    config
) -> Tuple[HeteroData, Dict, Dict]:
    """
    Build PyTorch Geometric HeteroData object.
    
    Returns:
        - HeteroData: graph object
        - user_id_to_index: mapping dict
        - place_id_to_index: mapping dict
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
```

### 3.4 Graph Serialization

**Save graph for training**:

```python
# File: recsys/features/graph_builder.py

def save_graph(
    data: HeteroData,
    user_mappings: Tuple[Dict, Dict],
    place_mappings: Tuple[Dict, Dict],
    output_dir: str
):
    """
    Serialize graph and mappings to disk.
    """
    import pickle
    
    # Save graph
    torch.save(data, f"{output_dir}/hetero_graph.pt")
    
    # Save mappings
    user_id_to_index, index_to_user_id = user_mappings
    place_id_to_index, index_to_place_id = place_mappings
    
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
```

### 3.5 Integration Contract with LLD

**CRITICAL**: The following must match between GNN training and serving:

1. **Feature dimensions**:
   - `D_user_raw = 148`
   - `D_place_raw = 114`
   - `D_edge_up = 12`
   - `D_edge_uu = 3`

2. **Normalization schemes**:
   - All preference vectors sum to 1.0
   - Behavioral stats normalized to [0, 1] ranges
   - Popularity metrics log-normalized

3. **Categorical encodings**:
   - City IDs: 0 to `N_CITIES-1`
   - Neighborhood IDs: 0 to `N_NEIGHBORHOODS_PER_CITY-1` (per city)
   - Price bands: 0 to 4
   - Time slots: 0 to 5

4. **ID mappings**:
   - Must be persisted and loaded consistently
   - User/place IDs in serving layer must map to same graph indices used in training

---

## 4. GNN Model Architecture (PyTorch Implementation)

### 4.1 Model Configuration

**File**: `recsys/config/model_config.py`

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Hyperparameters for GNN model.
    MUST be consistent between training and serving.
    """
    # Embedding dimension
    D_MODEL = 128  # Output dimension of encoders and GNN layers
    
    # Encoder architecture
    CITY_EMBED_DIM = 16
    NEIGHBORHOOD_EMBED_DIM = 32
    PRICE_EMBED_DIM = 8
    TIME_SLOT_EMBED_DIM = 8
    
    # GNN backbone
    NUM_GNN_LAYERS = 2  # Number of message passing layers
    GNN_HIDDEN_DIM = 128  # Hidden dimension in conv layers
    GNN_DROPOUT = 0.1
    GNN_AGGR = 'mean'  # Aggregation: 'mean', 'sum', or 'attention'
    
    # Task heads
    PLACE_HEAD_HIDDEN = [256, 128]  # Hidden layers for place head MLP
    FRIEND_HEAD_HIDDEN = [256, 128]  # Hidden layers for friend head MLP
    HEAD_DROPOUT = 0.2
    
    # Context vector dimensions
    D_CTX_PLACE = 16  # Dimension of place context vector
    D_CTX_FRIEND = 16  # Dimension of friend context vector
    
    # Training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE_PLACE = 512
    BATCH_SIZE_FRIEND = 512
    
    # Loss weights
    LAMBDA_PLACE = 1.0
    LAMBDA_FRIEND = 0.5
    LAMBDA_ATTEND = 0.3
```

### 4.2 Input Feature Encoders

**File**: `recsys/ml/models/encoders.py`

```python
import torch
import torch.nn as nn
from typing import Dict

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
```

### 4.3 GNN Backbone Implementation

**File**: `recsys/ml/models/backbone.py`

```python
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from torch_geometric.data import HeteroData
from typing import Dict

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
```

### 4.3 Task Heads

We add two separate heads on top of the shared backbone.

#### 4.3.1 Place Recommendation Head

- Inputs:
  - User embedding: \\(z_u\\).
  - Place embedding: \\(z_p\\).
  - Context vector \\(c_\text{ctx}\\), encoding:
    - Query city (if specified).
    - Desired time slot (e.g., tonight, weekend).
    - Optional desired coarse category and/or fine tags (mood).

- Construct feature vector:

\\[
f_{up} = [ z_u, z_p, z_u \odot z_p, |z_u - z_p|, c_\text{ctx} ]
\\]

- Pass through an MLP:

\\[
s_\text{place} = \text{MLP}_\text{place}(f_{up}) \in \mathbb{R}
\\]

This scalar is the **relevance score** of place \\(p\\) for user \\(u\\) under context \\(c_\text{ctx}\\).

#### 4.3.2 Friend / People Compatibility Head

- Inputs:
  - Query user embedding \\(z_u\\).
  - Candidate user embedding \\(z_v\\).
  - Optional context \\(c_\text{friend}\\):
    - Candidate place or activity tag.
    - City, desired time slot, etc.

- Construct feature vector:

\\[
f_{uv} = [ z_u, z_v, z_u \odot z_v, |z_u - z_v|, c_\text{friend} ]
\\]

- Two outputs:
  - **Compatibility score**:

    \\[
    s_\text{compat} = \text{MLP}_\text{compat}(f_{uv})
    \\]

  - **Attendance / acceptance probability**:

    \\[
    p_\text{attend} = \sigma(\text{MLP}_\text{attend}(f_{uv}))
    \\]

- Combined score (for ranking):

\\[
s_\text{final} = \alpha \, s_\text{compat} + (1 - \alpha) \, p_\text{attend}
\\]

with \\(\alpha \in [0, 1]\\) a tunable hyperparameter.

---

## 5. Training Objectives and Losses

We jointly train the model on:

- **Place recommendation task** (user–place).
- **Friend / people compatibility task** (user–user).

### 5.1 Place Recommendation Loss

- **Positive samples**:
  - User–place interactions where `implicit_rating` is high:
    - Strong behavioral signals like:
      - High dwell time.
      - Likes/saves.
      - Attended flag.

- **Negative samples**:
  - Places that the user has **not** interacted with.
  - To make the task meaningful:
    - Sample negatives within the **same city**.
    - Optionally within the same **coarse category**.

- **Loss options**:
  - **BPR (Bayesian Personalized Ranking) loss**:

    For each triplet \\((u, p^+, p^-)\\):

    \\[
    \mathcal{L}_\text{place} = -\log \sigma( s_\text{place}(u, p^+) - s_\text{place}(u, p^-) )
    \\]

  - Or **binary cross-entropy** on `(u, p)` pairs, with labels 1 (positive) / 0 (negative).

### 5.2 Friend / People Compatibility Loss

- **Positive pairs** `(u, v)`:
  - Users who:
    - Have co-attended events/places (synthetic or real).
    - Or have high social edge strength.

- **Negative pairs** `(u, v)`:
  - Users in the same city with:
    - Low interest overlap.
    - No co-attendance.

- **Compatibility loss**:

  - Binary cross-entropy on logits:

  \\[
  \mathcal{L}_\text{compat} = \text{BCEWithLogits}( s_\text{compat}(u, v), \text{label}_\text{compat} )
  \\]

- **Attendance loss**:

  - If we simulate or observe acceptance/attendance labels:

  \\[
  \mathcal{L}_\text{attend} = \text{BCE}( p_\text{attend}(u, v), \text{label}_\text{attend} )
  \\]

### 5.3 Joint Loss

The total loss combines both tasks (plus regularization):

\\[
\mathcal{L}_\text{total}
 = \lambda_\text{place} \, \mathcal{L}_\text{place}
 + \lambda_\text{friend} \, \mathcal{L}_\text{compat}
 + \lambda_\text{attend} \, \mathcal{L}_\text{attend}
 + \text{regularization}
\\]

where \\(\lambda_\text{place}, \lambda_\text{friend}, \lambda_\text{attend}\\) control the relative importance of each component.

---

## 6. Inference and Serving Strategy

The trained model is used primarily to produce **embeddings** and train **task heads**:

- Offline:
  - Compute **final user embeddings** `z_u` and **place embeddings** `z_p`.
  - Persist them to storage.
- Online:
  - Use **approximate nearest neighbor (ANN)** indices to retrieve candidates.
  - Re-score candidates with the **task heads** for each request.

### 6.1 Embedding Export

After training is complete:

1. Run a full forward pass on the graph with the trained model.
2. Obtain:
   - `z_user[user_id]` for all users.
   - `z_place[place_id]` for all places.
3. Save embeddings to disk (e.g., Parquet/CSV) or database.
4. Store mappings between internal indices and `user_id`/`place_id`.

### 6.2 ANN Indexes

To achieve **sub-second latency**, build separate ANN indexes:

- **Place index**:
  - For each city, build an ANN index over all place embeddings belonging to that city.
  - Metric: typically inner product or cosine similarity.

- **User index**:
  - For each city, build an ANN index over user embeddings in that city.

These indexes are loaded into memory by the online service.

### 6.3 Online Place Recommendation Flow

Given a request:

- Inputs:
  - `user_id`
  - Optional `city_id` (or derived from user).
  - Optional context (time slot, mood tags).

Steps:

1. **Fetch user embedding**:
   - Look up `z_u` from the embedding store.
2. **Determine candidate city**:
   - Use provided `city_id` or user’s home / current city.
3. **Candidate retrieval**:
   - Query the **place ANN index** for that city using `z_u`.
   - Retrieve top \\(M\\) candidates (e.g., \\(M = 200\\)).
4. **Scoring**:
   - For each candidate place:
     - Retrieve `z_p`.
     - Build context vector `c_ctx`.
     - Compute `s_place(u, p | ctx)` via the **place head**.
5. **Ranking and filtering**:
   - Sort candidates by score.
   - Apply hard filters (price, category, etc. if requested).
6. **Explainability**:
   - For the final top \\(K\\), generate explanation strings (see Section 8).
7. **Return results**:
   - Place IDs, scores, and explanations.

### 6.4 Online People Recommendation Flow

Given a request:

- Inputs:
  - `user_id`.
  - Optional `city_id`.
  - Optional `target_place_id` or `activity_tags`.

Steps:

1. **Fetch query embedding** `z_u`.
2. **Determine candidate city**.
3. **Candidate retrieval**:
   - Query the **user ANN index** for that city using `z_u`.
   - Exclude self and blocked users.
4. **Scoring**:
   - For each candidate user `v`:
     - Fetch `z_v`.
     - Build context `c_friend` (including activity/place if provided).
     - Compute `s_compat(u, v | ctx)` and `p_attend(u, v | ctx)`.
     - Combine into `s_final`.
5. **Ranking**:
   - Sort by `s_final`.
6. **Explainability**:
   - Generate explanations for top \\(K\\) matches:
     - Overlapping tags, similar neighborhoods, etc.
7. **Return results**:
   - Candidate user IDs, scores, attendance probability, explanations.

---

## 7. Synthetic Data Generation Plan

The goal is to generate a **synthetic but realistic world** that:

- Matches the graph schema (users, places, interactions, social edges).
- Encodes plausible behavior patterns and preferences.
- Provides labels for both:
  - **Place recommendation**.
  - **Friend / people compatibility**.

### 7.1 Global Configuration

- **Counts**:
  - `N_users = 10_000`.
  - `N_places = 10_000`.
  - `N_cities` ≈ 5–10.
  - Neighborhoods per city: 10–20.
- **Tags**:
  - `C_coarse` ≈ 5–8 coarse categories.
  - `C_fine` ≈ 100 fine tags.
  - `C_vibe` ≈ 30 vibe/personality tags.

### 7.2 Synthetic Places

For each place:

1. **Location**:
   - Sample `city_id` using a skewed distribution (some cities larger).
   - Sample `neighborhood_id` uniformly or from city-specific distribution.
2. **Categories**:
   - Sample 1–2 coarse categories.
   - Construct `category_one_hot` (possibly multi-hot).
3. **Fine tags**:
   - For each coarse category, define a subset of consistent fine tags (e.g., sports → bouldering, badminton).
   - Sample 3–7 fine tags and assign random weights.
   - Normalize to get `fine_tag_vector`.
4. **Operational attributes**:
   - `price_band` sampled from a city-specific distribution.
   - `typical_time_slot` sampled from a categorical distribution (e.g., nightlife vs brunch).
5. **Popularity**:
   - Sample `base_popularity` from a log-normal distribution.
   - Derive `avg_daily_visits`, `conversion_rate`, `novelty_score` from `base_popularity` with noise.

### 7.3 Synthetic Users

For each user:

1. **Home location**:
   - Sample `home_city_id` with more users assigned to larger cities.
   - Sample `home_neighborhood_id` within that city.
2. **Interest over coarse categories**:
   - Sample `user_cat_pref ~ Dirichlet(alpha_cat)` (hyperparameters tuned to produce diverse but focused users).
3. **Fine-tag interest**:
   - For each category where user has higher weight, bias fine-tag probabilities towards tags associated with that category.
   - Sample `user_fine_pref ~ Dirichlet(alpha_fine_conditional)` and normalize.
4. **Vibe / personality profile**:
   - Choose 3–10 vibe tags as “core traits” for the user.
   - Assign random positive weights and normalize into `user_vibe_pref`.
5. **Behavior statistics**:
   - Sample:
     - `avg_sessions_per_week` from a small range (e.g., 1–10).
     - `avg_views_per_session` (e.g., 10–100).
     - `avg_likes_per_session`, `avg_saves_per_session`, `avg_attends_per_month`.
6. **Location behavior**:
   - Concentrate activity on 1–3 neighborhoods:
     - Sample a neighborhood subset.
     - Draw a Dirichlet over them for `area_freqs`.

### 7.4 Synthetic User–Place Interactions

We model interactions to be **consistent with**:

- User category and fine-tag preferences.
- User location behavior.
- Place categories, fine tags, and popularity.

For each user:

1. **Total interactions**:
   - Approximate `N_interactions` per user:
     - `N_interactions ≈ avg_sessions_per_week * time_horizon_weeks * views_per_session`.
     - Use a cap (e.g., 100–1000 per user).
2. **Candidate places**:
   - Start with places in the same city.
   - Occasionally include places in nearby or other cities to simulate travel.
3. **Preference score** for each candidate place `p`:
   - Compute:
     - **Interest similarity**:
       - `sim_interest = cosine(user_fine_pref, place_fine_tags)`.
     - **Category alignment**:
       - `sim_cat = dot(user_cat_pref, place_category_one_hot)`.
     - **Location factor**:
       - Higher for same neighborhood or frequent areas.
     - **Popularity factor**:
       - Derived from `base_popularity`.
   - Combine:

     \\[
     \text{score}_{up} = w_1 \cdot \text{sim}_\text{interest}
       + w_2 \cdot \text{sim}_\text{cat}
       + w_3 \cdot \text{loc\_factor}
       + w_4 \cdot \text{popularity}
       + \text{noise}
     \\]

4. **Sampling interactions**:
   - Convert scores to probabilities using softmax over candidate places.
   - Sample places according to these probabilities until `N_interactions` is reached.
5. **Event details per interaction**:
   - For each sampled `(u, p)`:
     - **Dwell time**:
       - Sample from a distribution increasing in `score_up`.
     - **Actions**:
       - With higher `score_up`, increase probability of like/save/attend.
       - Example:
         - Low score: view-only.
         - Medium: view + like.
         - High: view + like + save, and sometimes attend.
     - **Implicit rating**:
       - Map (dwell, like/save, attend) to rating in [1, 5] with added noise.
     - **Timestamp**:
       - Sample a datetime in a chosen time horizon.
       - Derive `time_of_day_bucket`, `day_of_week_bucket` from timestamp.

### 7.5 Synthetic User–User Edges and Friend Labels

We need:

- A graph of soft social edges.
- Labels for training the friend compatibility and attendance heads.

Steps:

1. **Interest similarity pre-graph**:
   - Use user fine-tag vectors (`user_fine_pref`) and possibly vibe vectors.
   - For each user:
     - Find top-k similar users via cosine similarity (ANN can be used even here).
   - Candidate social edges: pairs with similarity above a threshold.
2. **Co-attendance simulation**:
   - For some high-similarity pairs in the same city:
     - Simulate joint outings:
       - Choose a place where both have high `score_up`.
       - Mark as “co-attended”.
   - Maintain `co_attendance_count` for each pair.
3. **Social edge construction**:
   - For each candidate pair `(u, v)`:
     - Compute:
       - `interest_overlap_score` (cosine).
       - `co_attendance_count`.
       - `same_neighborhood_freq` (based on `area_freqs`).
   - Keep edges where the combined score exceeds a threshold.
4. **Friend labels**:
   - **Positive labels** for compatibility:
     - Pairs with high `interest_overlap_score` and/or `co_attendance_count`.
   - **Negative labels**:
     - Randomly sampled same-city pairs with low overlap and no co-attendance.
   - **Attendance labels** for the friend head:
     - For some positive pairs, simulate acceptance/attendance behavior:
       - Higher probability of accept/attend when similarity and co-attendance are high.
     - Record `label_attend` (0/1).

---

## 8. Explainability Signals (for Later Consumption)

To power user-facing explanations like “you both like fishing”:

- We rely on the **same features** used in the model:
  - Overlaps in **fine-grained interest tags**.
  - Overlaps in **coarse categories**.
  - Overlaps in **vibe/personality tags**.
  - Overlaps in **areas / neighborhoods**.
- For each recommendation (place or person), we can:
  - Identify the top overlapping features:
    - Top-k common tags by product of user and place/user weights.
    - Shared neighborhoods with high `area_freqs`.
  - Turn these into short templates:
    - **Place**:
      - “Matches your interest in `{tag1}` and `{tag2}`.”
      - “You often go out in `{neighborhood}`, and this place is nearby.”
    - **People**:
      - “You both like `{tag1}` and `{tag2}`.”
      - “You both often go out in `{neighborhood}`.”

These explanation primitives will be consumed by a separate explanation/service layer described in the LLD.


