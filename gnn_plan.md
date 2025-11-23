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

### 4.4 Task Heads Implementation

**File**: `recsys/ml/models/heads.py`

```python
import torch
import torch.nn as nn

class PlaceHead(nn.Module):
    """
    Scoring head for user-place recommendations.
    
    Takes user embedding, place embedding, and context.
    Outputs scalar relevance score.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimension: z_u + z_p + z_u*z_p + |z_u-z_p| + ctx
        # = D_MODEL + D_MODEL + D_MODEL + D_MODEL + D_CTX_PLACE
        # = 4 * D_MODEL + D_CTX_PLACE
        input_dim = 4 * config.D_MODEL + config.D_CTX_PLACE
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.PLACE_HEAD_HIDDEN:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.HEAD_DROPOUT)
            ])
            prev_dim = hidden_dim
        
        # Final scoring layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        z_user: torch.Tensor,
        z_place: torch.Tensor,
        ctx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_user: (batch_size, D_MODEL) user embeddings
            z_place: (batch_size, D_MODEL) place embeddings
            ctx: (batch_size, D_CTX_PLACE) context features
        
        Returns:
            (batch_size, 1) relevance scores
        """
        # Construct interaction features
        z_mul = z_user * z_place  # Element-wise product
        z_diff = torch.abs(z_user - z_place)  # Absolute difference
        
        # Concatenate all features
        features = torch.cat([z_user, z_place, z_mul, z_diff, ctx], dim=1)
        
        # Score
        score = self.mlp(features)  # (batch_size, 1)
        
        return score.squeeze(-1)  # (batch_size,)


class ContextEncoder(nn.Module):
    """
    Encodes context features for place/friend recommendations.
    """
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        
        # Embeddings for context
        self.city_embed = nn.Embedding(N_CITIES, 8)
        self.time_slot_embed = nn.Embedding(N_TIME_SLOTS, 8)
        
        # For optional desired category/tags (multi-hot input)
        # Input: C_COARSE + small MLP
        self.category_proj = nn.Linear(C_COARSE, output_dim // 2)
        
        # Combined projection
        self.proj = nn.Linear(8 + 8 + output_dim // 2, output_dim)
    
    def forward(
        self,
        city_id: torch.Tensor,
        time_slot: torch.Tensor,
        desired_categories: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            city_id: (batch_size,) city IDs
            time_slot: (batch_size,) time slot IDs
            desired_categories: (batch_size, C_COARSE) multi-hot desired categories
        
        Returns:
            (batch_size, output_dim) context vector
        """
        city_emb = self.city_embed(city_id)
        time_emb = self.time_slot_embed(time_slot)
        cat_emb = torch.relu(self.category_proj(desired_categories))
        
        combined = torch.cat([city_emb, time_emb, cat_emb], dim=1)
        ctx = self.proj(combined)
        
        return ctx


class FriendHead(nn.Module):
    """
    Scoring head for user-user compatibility.
    
    Outputs two scores:
    - Compatibility score (how well they match)
    - Attendance probability (likelihood to accept/attend)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimension
        input_dim = 4 * config.D_MODEL + config.D_CTX_FRIEND
        
        # Shared trunk
        trunk_layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.FRIEND_HEAD_HIDDEN[:-1]:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.HEAD_DROPOUT)
            ])
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Compatibility head (outputs logit)
        self.compat_head = nn.Sequential(
            nn.Linear(prev_dim, config.FRIEND_HEAD_HIDDEN[-1]),
            nn.ReLU(),
            nn.Linear(config.FRIEND_HEAD_HIDDEN[-1], 1)
        )
        
        # Attendance probability head (outputs logit, will apply sigmoid)
        self.attend_head = nn.Sequential(
            nn.Linear(prev_dim, config.FRIEND_HEAD_HIDDEN[-1]),
            nn.ReLU(),
            nn.Linear(config.FRIEND_HEAD_HIDDEN[-1], 1)
        )
    
    def forward(
        self,
        z_user_u: torch.Tensor,
        z_user_v: torch.Tensor,
        ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_user_u: (batch_size, D_MODEL) query user embeddings
            z_user_v: (batch_size, D_MODEL) candidate user embeddings
            ctx: (batch_size, D_CTX_FRIEND) context features
        
        Returns:
            compat_score: (batch_size,) compatibility scores (logits)
            attend_prob: (batch_size,) attendance probabilities (0-1)
        """
        # Construct interaction features
        z_mul = z_user_u * z_user_v
        z_diff = torch.abs(z_user_u - z_user_v)
        
        # Concatenate
        features = torch.cat([z_user_u, z_user_v, z_mul, z_diff, ctx], dim=1)
        
        # Shared representation
        shared = self.trunk(features)
        
        # Two heads
        compat_logits = self.compat_head(shared).squeeze(-1)  # (batch_size,)
        attend_logits = self.attend_head(shared).squeeze(-1)
        attend_prob = torch.sigmoid(attend_logits)  # (batch_size,)
        
        return compat_logits, attend_prob
    
    def compute_combined_score(
        self,
        compat_score: torch.Tensor,
        attend_prob: torch.Tensor,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Combine compatibility and attendance into final ranking score.
        
        Args:
            compat_score: Compatibility scores (logits or normalized)
            attend_prob: Attendance probabilities (0-1)
            alpha: Weight for compatibility (1-alpha for attendance)
        
        Returns:
            Combined scores
        """
        # Normalize compat_score to [0, 1] via sigmoid
        compat_norm = torch.sigmoid(compat_score)
        
        return alpha * compat_norm + (1 - alpha) * attend_prob
```

---

## 5. Training Pipeline (PyTorch Implementation)

### 5.1 Loss Functions

**File**: `recsys/ml/models/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.
    
    Args:
        pos_scores: (batch_size,) scores for positive items
        neg_scores: (batch_size,) scores for negative items
    
    Returns:
        Scalar loss
    """
    # Loss = -log(sigmoid(pos - neg))
    # = log(1 + exp(-(pos - neg)))
    # = softplus(neg - pos)
    loss = F.softplus(neg_scores - pos_scores).mean()
    return loss


def binary_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Binary cross-entropy loss with logits.
    
    Args:
        logits: (batch_size,) predicted logits
        labels: (batch_size,) binary labels (0 or 1)
    
    Returns:
        Scalar loss
    """
    return F.binary_cross_entropy_with_logits(logits, labels.float())


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task GNN training.
    """
    
    def __init__(
        self,
        lambda_place: float = 1.0,
        lambda_friend: float = 0.5,
        lambda_attend: float = 0.3
    ):
        super().__init__()
        self.lambda_place = lambda_place
        self.lambda_friend = lambda_friend
        self.lambda_attend = lambda_attend
    
    def forward(
        self,
        loss_place: torch.Tensor,
        loss_friend: torch.Tensor,
        loss_attend: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Combine losses with weights.
        
        Returns:
            total_loss: Weighted sum
            loss_dict: Individual loss values for logging
        """
        total_loss = (
            self.lambda_place * loss_place +
            self.lambda_friend * loss_friend +
            self.lambda_attend * loss_attend
        )
        
        loss_dict = {
            'loss_place': loss_place.item(),
            'loss_friend': loss_friend.item(),
            'loss_attend': loss_attend.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict
```

### 5.2 Dataset and Samplers

**File**: `recsys/ml/training/datasets.py`

```python
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
import random

class PlaceRecommendationDataset(Dataset):
    """
    Dataset for place recommendation task with BPR sampling.
    
    For each user-place positive interaction, samples a negative place.
    """
    
    def __init__(
        self,
        interactions: List[InteractionSchema],
        user_id_to_index: Dict[int, int],
        place_id_to_index: Dict[int, int],
        places: List[PlaceSchema],
        rating_threshold: float = 3.5,
        negatives_per_positive: int = 1
    ):
        """
        Args:
            interactions: All user-place interactions
            user_id_to_index: Mapping user_id -> graph index
            place_id_to_index: Mapping place_id -> graph index
            places: List of all places
            rating_threshold: Min implicit_rating to consider positive
            negatives_per_positive: How many negatives to sample per positive
        """
        self.user_id_to_index = user_id_to_index
        self.place_id_to_index = place_id_to_index
        
        # Filter positive interactions
        self.positives = [
            (inter.user_id, inter.place_id)
            for inter in interactions
            if inter.implicit_rating >= rating_threshold
        ]
        
        # Build user -> positive places mapping
        self.user_to_pos_places = {}
        for user_id, place_id in self.positives:
            if user_id not in self.user_to_pos_places:
                self.user_to_pos_places[user_id] = set()
            self.user_to_pos_places[user_id].add(place_id)
        
        # Build city -> places mapping for negative sampling
        self.city_to_places = {}
        for place in places:
            if place.city_id not in self.city_to_places:
                self.city_to_places[place.city_id] = []
            self.city_to_places[place.city_id].append(place.place_id)
        
        # Store user home cities
        # (Assume we have access to users list or can derive from interactions)
        self.user_to_city = {}  # To be filled externally
        
        self.negatives_per_positive = negatives_per_positive
    
    def __len__(self):
        return len(self.positives) * self.negatives_per_positive
    
    def __getitem__(self, idx):
        # Get positive sample
        pos_idx = idx // self.negatives_per_positive
        user_id, pos_place_id = self.positives[pos_idx]
        
        # Sample negative place from same city
        city_id = self.user_to_city.get(user_id, 0)  # Default to city 0
        candidate_places = self.city_to_places.get(city_id, list(self.place_id_to_index.keys()))
        
        # Exclude positives
        pos_set = self.user_to_pos_places.get(user_id, set())
        neg_candidates = [p for p in candidate_places if p not in pos_set]
        
        if len(neg_candidates) == 0:
            # Fallback to any place
            neg_candidates = [p for p in self.place_id_to_index.keys() if p not in pos_set]
        
        neg_place_id = random.choice(neg_candidates)
        
        # Convert to graph indices
        user_idx = self.user_id_to_index[user_id]
        pos_place_idx = self.place_id_to_index[pos_place_id]
        neg_place_idx = self.place_id_to_index[neg_place_id]
        
        return {
            'user_idx': user_idx,
            'pos_place_idx': pos_place_idx,
            'neg_place_idx': neg_place_idx
        }


class FriendCompatibilityDataset(Dataset):
    """
    Dataset for friend compatibility task.
    """
    
    def __init__(
        self,
        friend_labels: List[FriendLabelSchema],
        user_id_to_index: Dict[int, int]
    ):
        self.friend_labels = friend_labels
        self.user_id_to_index = user_id_to_index
    
    def __len__(self):
        return len(self.friend_labels)
    
    def __getitem__(self, idx):
        label = self.friend_labels[idx]
        
        user_u_idx = self.user_id_to_index[label.user_u]
        user_v_idx = self.user_id_to_index[label.user_v]
        
        return {
            'user_u_idx': user_u_idx,
            'user_v_idx': user_v_idx,
            'label_compat': label.label_compat,
            'label_attend': label.label_attend
        }
```

### 5.3 Training Script

**File**: `scripts/run_train_gnn.py`

```python
#!/usr/bin/env python3
"""
Main training script for GNN recommendation model.
"""

import torch
from torch.utils.data import DataLoader
from recsys.config import ModelConfig
from recsys.features.graph_builder import load_graph
from recsys.data.repositories import *
from recsys.ml.models.encoders import UserEncoder, PlaceEncoder
from recsys.ml.models.backbone import GraphRecBackbone
from recsys.ml.models.heads import PlaceHead, FriendHead, ContextEncoder
from recsys.ml.training.datasets import PlaceRecommendationDataset, FriendCompatibilityDataset
from recsys.ml.training.train_loop import GNNTrainer
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
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
    place_loader = DataLoader(place_dataset, batch_size=config.BATCH_SIZE_PLACE, shuffle=True, num_workers=4)
    friend_loader = DataLoader(friend_dataset, batch_size=config.BATCH_SIZE_FRIEND, shuffle=True, num_workers=4)
    
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
        print(f"Epoch {epoch}: {losses}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    # Final save
    trainer.save_checkpoint(f"{args.output_dir}/final_model.pt")
    print("Training complete!")


if __name__ == '__main__':
    main()
```

---

## 6. Inference and Serving (FastAPI Implementation)

### 6.1 Embedding Export Script

**File**: `scripts/run_export_embeddings.py`

```python
#!/usr/bin/env python3
"""
Export trained embeddings to storage for serving.
"""

import torch
import pandas as pd
import numpy as np
import argparse
from recsys.features.graph_builder import load_graph
from recsys.ml.models.encoders import UserEncoder, PlaceEncoder
from recsys.ml.models.backbone import GraphRecBackbone
from recsys.config.model_config import ModelConfig


def export_embeddings(checkpoint_path: str, data_dir: str, output_dir: str):
    """
    Load trained model and export user/place embeddings.
    """
    # Load config and graph
    config = ModelConfig()
    graph, user_id_to_index, place_id_to_index, index_to_user_id, index_to_place_id = load_graph(data_dir)
    
    # Load trained model
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
    user_df.to_parquet(f"{output_dir}/user_embeddings.parquet")
    
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
    place_df.to_parquet(f"{output_dir}/place_embeddings.parquet")
    
    print(f"Exported {len(user_embeddings)} user embeddings")
    print(f"Exported {len(place_embeddings)} place embeddings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    export_embeddings(args.checkpoint, args.data_dir, args.output_dir)
```

### 6.2 ANN Index Implementation

**File**: `recsys/serving/ann_index.py`

```python
import faiss
import numpy as np
from typing import List, Tuple, Dict
import pickle


class AnnIndex:
    """
    Wrapper around Faiss for ANN search.
    """
    
    def __init__(self, dimension: int, metric: str = 'cosine'):
        """
        Args:
            dimension: Embedding dimension
            metric: 'cosine' or 'l2'
        """
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.ids = []  # Maps index position -> actual ID
    
    def build(self, embeddings: np.ndarray, ids: List[int]):
        """
        Build index from embeddings.
        
        Args:
            embeddings: (N, D) array
            ids: List of N IDs corresponding to embeddings
        """
        assert len(embeddings) == len(ids)
        self.ids = ids
        
        # Normalize if cosine
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: (D,) query vector
            top_k: Number of results
        
        Returns:
            List of (id, distance) tuples
        """
        if self.index is None:
            return []
        
        query = query.reshape(1, -1).astype(np.float32)
        
        if self.metric == 'cosine':
            faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.ids):
                results.append((self.ids[idx], float(dist)))
        
        return results
    
    def save(self, path: str):
        """Save index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'metric': self.metric,
                'ids': self.ids,
                'index': faiss.serialize_index(self.index)
            }, f)
    
    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.metric = data['metric']
        self.ids = data['ids']
        self.index = faiss.deserialize_index(data['index'])


class CityAnnIndexManager:
    """
    Manages separate ANN indices per city.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.city_indices: Dict[int, AnnIndex] = {}
    
    def build_city_index(
        self,
        city_id: int,
        embeddings: np.ndarray,
        ids: List[int]
    ):
        """Build index for a specific city."""
        index = AnnIndex(self.dimension, metric='cosine')
        index.build(embeddings, ids)
        self.city_indices[city_id] = index
    
    def search(
        self,
        city_id: int,
        query: np.ndarray,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Search within a city's index."""
        if city_id not in self.city_indices:
            return []
        return self.city_indices[city_id].search(query, top_k)
    
    def save(self, output_dir: str, prefix: str):
        """Save all city indices."""
        for city_id, index in self.city_indices.items():
            index.save(f"{output_dir}/{prefix}_city_{city_id}.idx")
    
    def load(self, input_dir: str, prefix: str, city_ids: List[int]):
        """Load city indices."""
        for city_id in city_ids:
            path = f"{input_dir}/{prefix}_city_{city_id}.idx"
            try:
                index = AnnIndex(self.dimension)
                index.load(path)
                self.city_indices[city_id] = index
            except FileNotFoundError:
                print(f"Warning: Index for city {city_id} not found")
```

### 6.3 FastAPI Schemas

**File**: `recsys/serving/api_schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional


class PlaceRecommendationRequest(BaseModel):
    """Request for place recommendations."""
    user_id: int
    city_id: Optional[int] = None
    time_slot: Optional[int] = None  # 0-5
    desired_categories: Optional[List[int]] = None  # Indices into C_COARSE
    top_k: int = 10


class PlaceRecommendation(BaseModel):
    """Single place recommendation."""
    place_id: int
    score: float
    explanations: List[str]


class PlaceRecommendationResponse(BaseModel):
    """Response with place recommendations."""
    recommendations: List[PlaceRecommendation]


class PeopleRecommendationRequest(BaseModel):
    """Request for people recommendations."""
    user_id: int
    city_id: Optional[int] = None
    target_place_id: Optional[int] = None
    activity_tags: Optional[List[int]] = None
    top_k: int = 10


class PeopleRecommendation(BaseModel):
    """Single person recommendation."""
    user_id: int
    compat_score: float
    attend_prob: float
    combined_score: float
    explanations: List[str]


class PeopleRecommendationResponse(BaseModel):
    """Response with people recommendations."""
    recommendations: List[PeopleRecommendation]
```

### 6.4 Recommender Core (Business Logic)

**File**: `recsys/serving/recommender_core.py`

```python
import torch
import numpy as np
from typing import List, Dict, Tuple
from recsys.serving.ann_index import CityAnnIndexManager
from recsys.data.schemas import UserSchema, PlaceSchema
from recsys.ml.models.heads import PlaceHead, FriendHead, ContextEncoder
from recsys.config.model_config import ModelConfig


class EmbeddingStore:
    """
    In-memory storage for embeddings.
    """
    
    def __init__(self):
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.place_embeddings: Dict[int, np.ndarray] = {}
    
    def load_from_parquet(self, user_path: str, place_path: str):
        """Load embeddings from parquet files."""
        import pandas as pd
        
        user_df = pd.read_parquet(user_path)
        for _, row in user_df.iterrows():
            self.user_embeddings[row['user_id']] = np.array(row['embedding'])
        
        place_df = pd.read_parquet(place_path)
        for _, row in place_df.iterrows():
            self.place_embeddings[row['place_id']] = np.array(row['embedding'])
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        return self.user_embeddings.get(user_id)
    
    def get_place_embedding(self, place_id: int) -> np.ndarray:
        return self.place_embeddings.get(place_id)


class PlaceRecommender:
    """
    Core logic for place recommendations.
    """
    
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        place_ann_manager: CityAnnIndexManager,
        place_head: PlaceHead,
        ctx_encoder: ContextEncoder,
        user_repo,
        place_repo,
        explanation_service,
        config: ModelConfig
    ):
        self.embedding_store = embedding_store
        self.place_ann = place_ann_manager
        self.place_head = place_head
        self.ctx_encoder = ctx_encoder
        self.user_repo = user_repo
        self.place_repo = place_repo
        self.explanation_service = explanation_service
        self.config = config
        
        self.place_head.eval()
    
    def recommend(
        self,
        user_id: int,
        city_id: Optional[int] = None,
        time_slot: Optional[int] = None,
        desired_categories: Optional[List[int]] = None,
        top_k: int = 10,
        top_m_candidates: int = 200
    ) -> List[Dict]:
        """
        Generate place recommendations.
        
        Returns:
            List of dicts with place_id, score, explanations
        """
        # 1. Get user embedding
        z_u = self.embedding_store.get_user_embedding(user_id)
        if z_u is None:
            return []
        
        # 2. Determine city
        if city_id is None:
            user = self.user_repo.get_user(user_id)
            city_id = user.home_city_id
        
        # 3. ANN candidate retrieval
        candidates = self.place_ann.search(city_id, z_u, top_m_candidates)
        if not candidates:
            return []
        
        # 4. Prepare context
        if time_slot is None:
            time_slot = 3  # Default: evening
        
        if desired_categories is None:
            desired_categories = [0.0] * 6
        else:
            # Convert to multi-hot
            cat_vec = [0.0] * 6
            for cat_idx in desired_categories:
                if 0 <= cat_idx < 6:
                    cat_vec[cat_idx] = 1.0
            desired_categories = cat_vec
        
        # 5. Score with head
        scored_candidates = []
        
        with torch.no_grad():
            z_u_torch = torch.tensor(z_u, dtype=torch.float32).unsqueeze(0)
            
            for place_id, ann_score in candidates:
                z_p = self.embedding_store.get_place_embedding(place_id)
                if z_p is None:
                    continue
                
                z_p_torch = torch.tensor(z_p, dtype=torch.float32).unsqueeze(0)
                
                # Build context
                city_tensor = torch.tensor([city_id], dtype=torch.long)
                time_tensor = torch.tensor([time_slot], dtype=torch.long)
                cat_tensor = torch.tensor([desired_categories], dtype=torch.float32)
                
                ctx = self.ctx_encoder(city_tensor, time_tensor, cat_tensor)
                
                # Score
                score = self.place_head(z_u_torch, z_p_torch, ctx).item()
                
                scored_candidates.append((place_id, score))
        
        # 6. Sort and top-K
        scored_candidates.sort(key=lambda x: -x[1])
        top_candidates = scored_candidates[:top_k]
        
        # 7. Generate explanations
        results = []
        user = self.user_repo.get_user(user_id)
        
        for place_id, score in top_candidates:
            place = self.place_repo.get_place(place_id)
            explanations = self.explanation_service.explain_place(user, place)
            
            results.append({
                'place_id': place_id,
                'score': score,
                'explanations': explanations
            })
        
        return results


class PeopleRecommender:
    """
    Core logic for people recommendations.
    """
    
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        user_ann_manager: CityAnnIndexManager,
        friend_head: FriendHead,
        ctx_encoder: ContextEncoder,
        user_repo,
        explanation_service,
        config: ModelConfig
    ):
        self.embedding_store = embedding_store
        self.user_ann = user_ann_manager
        self.friend_head = friend_head
        self.ctx_encoder = ctx_encoder
        self.user_repo = user_repo
        self.explanation_service = explanation_service
        self.config = config
        
        self.friend_head.eval()
    
    def recommend(
        self,
        user_id: int,
        city_id: Optional[int] = None,
        target_place_id: Optional[int] = None,
        activity_tags: Optional[List[int]] = None,
        top_k: int = 10,
        top_m_candidates: int = 200,
        alpha: float = 0.7
    ) -> List[Dict]:
        """
        Generate people recommendations.
        
        Returns:
            List of dicts with user_id, compat_score, attend_prob, combined_score, explanations
        """
        # 1. Get query user embedding
        z_u = self.embedding_store.get_user_embedding(user_id)
        if z_u is None:
            return []
        
        # 2. Determine city
        if city_id is None:
            user = self.user_repo.get_user(user_id)
            city_id = user.home_city_id
        
        # 3. ANN retrieval
        candidates = self.user_ann.search(city_id, z_u, top_m_candidates)
        # Filter out self
        candidates = [(uid, score) for uid, score in candidates if uid != user_id]
        
        if not candidates:
            return []
        
        # 4. Prepare context (dummy for now)
        # Can be enhanced with target_place_id and activity_tags
        
        # 5. Score with friend head
        scored_candidates = []
        
        with torch.no_grad():
            z_u_torch = torch.tensor(z_u, dtype=torch.float32).unsqueeze(0)
            
            for candidate_uid, ann_score in candidates:
                z_v = self.embedding_store.get_user_embedding(candidate_uid)
                if z_v is None:
                    continue
                
                z_v_torch = torch.tensor(z_v, dtype=torch.float32).unsqueeze(0)
                
                # Dummy context
                ctx = torch.zeros(1, self.config.D_CTX_FRIEND)
                
                # Score
                compat_logit, attend_prob = self.friend_head(z_u_torch, z_v_torch, ctx)
                
                compat_score = torch.sigmoid(compat_logit).item()
                attend_prob = attend_prob.item()
                
                combined_score = alpha * compat_score + (1 - alpha) * attend_prob
                
                scored_candidates.append((
                    candidate_uid, compat_score, attend_prob, combined_score
                ))
        
        # 6. Sort and top-K
        scored_candidates.sort(key=lambda x: -x[3])
        top_candidates = scored_candidates[:top_k]
        
        # 7. Generate explanations
        results = []
        query_user = self.user_repo.get_user(user_id)
        
        for candidate_uid, compat_score, attend_prob, combined_score in top_candidates:
            candidate_user = self.user_repo.get_user(candidate_uid)
            explanations = self.explanation_service.explain_people(query_user, candidate_user)
            
            results.append({
                'user_id': candidate_uid,
                'compat_score': compat_score,
                'attend_prob': attend_prob,
                'combined_score': combined_score,
                'explanations': explanations
            })
        
        return results
```

### 6.5 Explanation Service

**File**: `recsys/serving/explanations.py`

```python
from typing import List
from recsys.data.schemas import UserSchema, PlaceSchema
from recsys.config.constants import FINE_TAGS, COARSE_CATEGORIES, VIBE_TAGS
import numpy as np


class ExplanationService:
    """
    Generates human-readable explanations for recommendations.
    """
    
    def explain_place(
        self,
        user: UserSchema,
        place: PlaceSchema,
        top_k_tags: int = 2
    ) -> List[str]:
        """
        Generate explanations for why a place was recommended to a user.
        
        Args:
            user: UserSchema
            place: PlaceSchema
            top_k_tags: Number of tag overlaps to mention
        
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # 1. Find overlapping fine tags
        user_fine = np.array(user.fine_pref)
        place_fine = np.array(place.fine_tag_vector)
        
        # Element-wise product to find mutual high-weight tags
        overlap_scores = user_fine * place_fine
        top_indices = np.argsort(-overlap_scores)[:top_k_tags]
        
        top_tags = [FINE_TAGS[idx] for idx in top_indices if overlap_scores[idx] > 0.01]
        
        if len(top_tags) >= 2:
            explanations.append(
                f"Matches your interest in {top_tags[0]} and {top_tags[1]}."
            )
        elif len(top_tags) == 1:
            explanations.append(
                f"Matches your interest in {top_tags[0]}."
            )
        
        # 2. Check coarse category alignment
        user_cat = np.array(user.cat_pref)
        place_cat = np.array(place.category_one_hot)
        
        top_cat_idx = np.argmax(user_cat * place_cat)
        if user_cat[top_cat_idx] > 0.15 and place_cat[top_cat_idx] > 0:
            category_name = COARSE_CATEGORIES[top_cat_idx]
            explanations.append(
                f"You enjoy {category_name} spots."
            )
        
        # 3. Check neighborhood proximity
        if place.neighborhood_id in user.area_freqs:
            explanations.append(
                f"You often go out in this neighborhood."
            )
        
        # If no strong explanations, add a generic one
        if not explanations:
            explanations.append("Recommended based on your activity history.")
        
        return explanations[:3]  # Max 3 explanations
    
    def explain_people(
        self,
        user_u: UserSchema,
        user_v: UserSchema,
        top_k_tags: int = 2
    ) -> List[str]:
        """
        Generate explanations for why two users are compatible.
        
        Args:
            user_u: Query user
            user_v: Candidate user
            top_k_tags: Number of overlaps to mention
        
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # 1. Find overlapping vibe/personality tags
        vibe_u = np.array(user_u.vibe_pref)
        vibe_v = np.array(user_v.vibe_pref)
        
        vibe_overlap = vibe_u * vibe_v
        top_vibe_indices = np.argsort(-vibe_overlap)[:top_k_tags]
        
        top_vibe_tags = [
            VIBE_TAGS[idx] for idx in top_vibe_indices
            if vibe_overlap[idx] > 0.01
        ]
        
        if len(top_vibe_tags) >= 2:
            explanations.append(
                f"You both are {top_vibe_tags[0]} and {top_vibe_tags[1]}."
            )
        elif len(top_vibe_tags) == 1:
            explanations.append(
                f"You both are {top_vibe_tags[0]}."
            )
        
        # 2. Find overlapping fine interests
        fine_u = np.array(user_u.fine_pref)
        fine_v = np.array(user_v.fine_pref)
        
        fine_overlap = fine_u * fine_v
        top_fine_indices = np.argsort(-fine_overlap)[:top_k_tags]
        
        top_fine_tags = [
            FINE_TAGS[idx] for idx in top_fine_indices
            if fine_overlap[idx] > 0.01
        ]
        
        if len(top_fine_tags) >= 2:
            explanations.append(
                f"You both like {top_fine_tags[0]} and {top_fine_tags[1]}."
            )
        elif len(top_fine_tags) == 1:
            explanations.append(
                f"You both like {top_fine_tags[0]}."
            )
        
        # 3. Check shared neighborhoods
        shared_neighborhoods = set(user_u.area_freqs.keys()) & set(user_v.area_freqs.keys())
        if shared_neighborhoods:
            explanations.append(
                "You both often go out in the same neighborhoods."
            )
        
        # Fallback
        if not explanations:
            explanations.append("You have similar interests and activity patterns.")
        
        return explanations[:3]
```

### 6.6 FastAPI Application

**File**: `recsys/serving/api_main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

from recsys.serving.api_schemas import *
from recsys.serving.recommender_core import PlaceRecommender, PeopleRecommender, EmbeddingStore
from recsys.serving.ann_index import CityAnnIndexManager
from recsys.serving.explanations import ExplanationService
from recsys.ml.models.heads import PlaceHead, FriendHead, ContextEncoder
from recsys.data.repositories import UserRepository, PlaceRepository
from recsys.config.model_config import ModelConfig
from recsys.config.constants import N_CITIES

# Initialize FastAPI app
app = FastAPI(
    title="Social Outing Recommender API",
    description="GNN-powered place and people recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (initialized on startup)
place_recommender = None
people_recommender = None


@app.on_event("startup")
async def startup_event():
    """
    Load models, embeddings, and indices on startup.
    """
    global place_recommender, people_recommender
    
    print("Loading configuration...")
    config = ModelConfig()
    
    # Load embeddings
    print("Loading embeddings...")
    embedding_store = EmbeddingStore()
    embedding_store.load_from_parquet(
        user_path="data/embeddings/user_embeddings.parquet",
        place_path="data/embeddings/place_embeddings.parquet"
    )
    
    # Load ANN indices
    print("Loading ANN indices...")
    place_ann = CityAnnIndexManager(dimension=config.D_MODEL)
    user_ann = CityAnnIndexManager(dimension=config.D_MODEL)
    
    city_ids = list(range(N_CITIES))
    place_ann.load("data/indices", "place", city_ids)
    user_ann.load("data/indices", "user", city_ids)
    
    # Load trained model heads
    print("Loading model heads...")
    checkpoint = torch.load("data/models/final_model.pt", map_location='cpu')
    
    place_head = PlaceHead(config)
    friend_head = FriendHead(config)
    place_ctx_encoder = ContextEncoder(config.D_CTX_PLACE)
    friend_ctx_encoder = ContextEncoder(config.D_CTX_FRIEND)
    
    place_head.load_state_dict(checkpoint['place_head'])
    friend_head.load_state_dict(checkpoint['friend_head'])
    place_ctx_encoder.load_state_dict(checkpoint['place_ctx_encoder'])
    friend_ctx_encoder.load_state_dict(checkpoint['friend_ctx_encoder'])
    
    place_head.eval()
    friend_head.eval()
    
    # Initialize repositories
    user_repo = UserRepository("data")
    place_repo = PlaceRepository("data")
    
    # Initialize explanation service
    explanation_service = ExplanationService()
    
    # Initialize recommenders
    place_recommender = PlaceRecommender(
        embedding_store=embedding_store,
        place_ann_manager=place_ann,
        place_head=place_head,
        ctx_encoder=place_ctx_encoder,
        user_repo=user_repo,
        place_repo=place_repo,
        explanation_service=explanation_service,
        config=config
    )
    
    people_recommender = PeopleRecommender(
        embedding_store=embedding_store,
        user_ann_manager=user_ann,
        friend_head=friend_head,
        ctx_encoder=friend_ctx_encoder,
        user_repo=user_repo,
        explanation_service=explanation_service,
        config=config
    )
    
    print("Server ready!")


@app.get("/")
async def root():
    return {"message": "Social Outing Recommender API", "status": "online"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/recommend/places", response_model=PlaceRecommendationResponse)
async def recommend_places(request: PlaceRecommendationRequest):
    """
    Get place recommendations for a user.
    
    Example request:
    ```
    {
      "user_id": 42,
      "city_id": 2,
      "time_slot": 3,
      "desired_categories": [0, 2],
      "top_k": 10
    }
    ```
    """
    try:
        results = place_recommender.recommend(
            user_id=request.user_id,
            city_id=request.city_id,
            time_slot=request.time_slot,
            desired_categories=request.desired_categories,
            top_k=request.top_k
        )
        
        recommendations = [
            PlaceRecommendation(**result) for result in results
        ]
        
        return PlaceRecommendationResponse(recommendations=recommendations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/people", response_model=PeopleRecommendationResponse)
async def recommend_people(request: PeopleRecommendationRequest):
    """
    Get people recommendations for a user.
    
    Example request:
    ```
    {
      "user_id": 42,
      "city_id": 2,
      "target_place_id": 1234,
      "top_k": 10
    }
    ```
    """
    try:
        results = people_recommender.recommend(
            user_id=request.user_id,
            city_id=request.city_id,
            target_place_id=request.target_place_id,
            activity_tags=request.activity_tags,
            top_k=request.top_k
        )
        
        recommendations = [
            PeopleRecommendation(**result) for result in results
        ]
        
        return PeopleRecommendationResponse(recommendations=recommendations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
```

**Run script**: `scripts/run_api_server.sh`

```bash
#!/bin/bash
# Start the FastAPI server

python -m uvicorn recsys.serving.api_main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

**Example API Usage**:

```python
import requests

# Place recommendations
response = requests.post(
    "http://localhost:8000/recommend/places",
    json={
        "user_id": 42,
        "city_id": 2,
        "time_slot": 3,  # evening
        "desired_categories": [0, 2],  # entertainment + clubs
        "top_k": 10
    }
)
places = response.json()["recommendations"]

# People recommendations
response = requests.post(
    "http://localhost:8000/recommend/people",
    json={
        "user_id": 42,
        "city_id": 2,
        "top_k": 10
    }
)
people = response.json()["recommendations"]
```

---

## 7. Synthetic Data Generation (Complete Python Implementation)

The goal is to generate a **synthetic but realistic world** that:

- Matches the graph schema (users, places, interactions, social edges).
- Encodes plausible behavior patterns and preferences.
- Provides labels for both:
  - **Place recommendation**.
  - **Friend / people compatibility**.

### 7.1 Generator Configuration

**File**: `recsys/synthetic/generator_config.py`

```python
"""
Configuration for synthetic data generation.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    
    # Scale
    N_USERS: int = 10_000
    N_PLACES: int = 10_000
    N_CITIES: int = 8
    N_NEIGHBORHOODS_PER_CITY: int = 15
    
    # Taxonomy dimensions (must match constants.py)
    C_COARSE: int = 6
    C_FINE: int = 100
    C_VIBE: int = 30
    
    # Category-to-tag mapping (which fine tags belong to which coarse categories)
    CATEGORY_TO_TAGS: Dict[int, List[int]] = None
    
    # Distribution parameters
    DIRICHLET_ALPHA_CAT: float = 2.0  # For category preferences
    DIRICHLET_ALPHA_FINE: float = 1.0  # For fine-tag preferences
    DIRICHLET_ALPHA_VIBE: float = 1.5  # For vibe preferences
    DIRICHLET_ALPHA_AREA: float = 3.0  # For area frequencies
    
    # City distribution (weights for sampling cities, sums to 1.0)
    CITY_WEIGHTS: List[float] = None
    
    # User behavior ranges
    SESSIONS_PER_WEEK_RANGE: tuple = (1.0, 8.0)
    VIEWS_PER_SESSION_RANGE: tuple = (10.0, 80.0)
    LIKES_PER_SESSION_RANGE: tuple = (0.5, 5.0)
    SAVES_PER_SESSION_RANGE: tuple = (0.2, 3.0)
    ATTENDS_PER_MONTH_RANGE: tuple = (1.0, 12.0)
    
    # Interaction parameters
    TIME_HORIZON_WEEKS: int = 12  # Simulate 12 weeks of data
    INTERACTIONS_PER_USER_RANGE: tuple = (50, 500)
    
    # Preference scoring weights
    W_INTEREST: float = 0.4
    W_CATEGORY: float = 0.3
    W_LOCATION: float = 0.2
    W_POPULARITY: float = 0.1
    
    # Social edge parameters
    SIMILARITY_THRESHOLD: float = 0.3  # Min cosine similarity for social edge
    K_SIMILAR_USERS: int = 50  # Find top-K similar users per user
    CO_ATTENDANCE_PROB: float = 0.1  # Probability of co-attendance for similar pairs
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    def __post_init__(self):
        """Initialize derived parameters."""
        if self.CITY_WEIGHTS is None:
            # Skewed distribution: some cities larger than others
            weights = np.random.dirichlet([2.0] * self.N_CITIES)
            self.CITY_WEIGHTS = weights.tolist()
        
        if self.CATEGORY_TO_TAGS is None:
            # Define which fine tags belong to which coarse categories
            tags_per_category = self.C_FINE // self.C_COARSE
            self.CATEGORY_TO_TAGS = {}
            for cat_idx in range(self.C_COARSE):
                start = cat_idx * tags_per_category
                end = start + tags_per_category
                if cat_idx == self.C_COARSE - 1:  # Last category gets remaining tags
                    end = self.C_FINE
                self.CATEGORY_TO_TAGS[cat_idx] = list(range(start, end))


def get_default_config() -> SyntheticConfig:
    """Get default configuration."""
    return SyntheticConfig()
```

### 7.2 Place Generation

**File**: `recsys/synthetic/generate_places.py`

```python
"""
Generate synthetic places.
"""

import numpy as np
from typing import List
from recsys.data.schemas import PlaceSchema
from recsys.synthetic.generator_config import SyntheticConfig


def generate_place(place_id: int, config: SyntheticConfig, rng: np.random.Generator) -> PlaceSchema:
    """
    Generate a single synthetic place.
    
    Args:
        place_id: Place ID (0 to N_PLACES-1)
        config: Generator configuration
        rng: Random number generator
    
    Returns:
        PlaceSchema
    """
    # 1. Location
    city_id = rng.choice(config.N_CITIES, p=config.CITY_WEIGHTS)
    neighborhood_id = rng.integers(0, config.N_NEIGHBORHOODS_PER_CITY)
    
    # 2. Categories (sample 1-2 coarse categories)
    num_categories = rng.choice([1, 2], p=[0.7, 0.3])  # 70% single category
    category_ids = rng.choice(
        config.C_COARSE,
        size=num_categories,
        replace=False
    ).tolist()
    
    # Create multi-hot encoding
    category_one_hot = [0.0] * config.C_COARSE
    for cat_id in category_ids:
        category_one_hot[cat_id] = 1.0 / num_categories  # Equal weight if multi-category
    
    # 3. Fine tags (sample from tags associated with selected categories)
    candidate_tags = []
    for cat_id in category_ids:
        candidate_tags.extend(config.CATEGORY_TO_TAGS[cat_id])
    
    # Sample 3-7 fine tags
    num_tags = rng.integers(3, 8)
    if len(candidate_tags) < num_tags:
        # If not enough candidate tags, add random tags
        other_tags = [t for t in range(config.C_FINE) if t not in candidate_tags]
        candidate_tags.extend(rng.choice(other_tags, size=num_tags - len(candidate_tags), replace=False))
    
    selected_tags = rng.choice(candidate_tags, size=min(num_tags, len(candidate_tags)), replace=False)
    
    # Assign weights and normalize
    tag_weights = rng.exponential(scale=1.0, size=len(selected_tags))
    fine_tag_vector = np.zeros(config.C_FINE)
    fine_tag_vector[selected_tags] = tag_weights
    fine_tag_vector = fine_tag_vector / fine_tag_vector.sum()  # Normalize
    
    # 4. Operational attributes
    # Price band: higher in certain cities (simulate expensive cities)
    expensive_cities = [0, 1]  # First two cities are expensive
    if city_id in expensive_cities:
        price_band = rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
    else:
        price_band = rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
    
    # Typical time slot (0-5)
    typical_time_slot = rng.integers(0, 6)
    
    # 5. Popularity (log-normal distribution)
    base_popularity = float(rng.lognormal(mean=1.0, sigma=1.0))
    
    # Derive related metrics with noise
    avg_daily_visits = base_popularity * rng.uniform(0.8, 1.2) * 10
    conversion_rate = min(0.95, max(0.05, rng.beta(2, 5) * (base_popularity / 10)))
    novelty_score = 1.0 / (1.0 + np.log1p(base_popularity))
    
    return PlaceSchema(
        place_id=place_id,
        city_id=int(city_id),
        neighborhood_id=int(neighborhood_id),
        category_ids=category_ids,
        category_one_hot=category_one_hot,
        fine_tag_vector=fine_tag_vector.tolist(),
        price_band=int(price_band),
        typical_time_slot=int(typical_time_slot),
        base_popularity=float(base_popularity),
        avg_daily_visits=float(avg_daily_visits),
        conversion_rate=float(conversion_rate),
        novelty_score=float(novelty_score)
    )


def generate_all_places(config: SyntheticConfig) -> List[PlaceSchema]:
    """
    Generate all synthetic places.
    
    Args:
        config: Generator configuration
    
    Returns:
        List of PlaceSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    
    places = []
    print(f"Generating {config.N_PLACES} places...")
    
    for place_id in range(config.N_PLACES):
        if (place_id + 1) % 1000 == 0:
            print(f"  Generated {place_id + 1}/{config.N_PLACES} places")
        
        place = generate_place(place_id, config, rng)
        places.append(place)
    
    print(f"✅ Generated {len(places)} places")
    return places
```

### 7.3 User Generation

**File**: `recsys/synthetic/generate_users.py`

```python
"""
Generate synthetic users.
"""

import numpy as np
from typing import List, Dict
from recsys.data.schemas import UserSchema
from recsys.synthetic.generator_config import SyntheticConfig


def generate_user(user_id: int, config: SyntheticConfig, rng: np.random.Generator) -> UserSchema:
    """
    Generate a single synthetic user.
    
    Args:
        user_id: User ID (0 to N_USERS-1)
        config: Generator configuration
        rng: Random number generator
    
    Returns:
        UserSchema
    """
    # 1. Home location (sample city according to weights)
    home_city_id = rng.choice(config.N_CITIES, p=config.CITY_WEIGHTS)
    home_neighborhood_id = rng.integers(0, config.N_NEIGHBORHOODS_PER_CITY)
    
    # 2. Interest over coarse categories (Dirichlet for diversity)
    cat_pref_raw = rng.dirichlet([config.DIRICHLET_ALPHA_CAT] * config.C_COARSE)
    cat_pref = cat_pref_raw / cat_pref_raw.sum()  # Ensure normalization
    
    # 3. Fine-tag interest (bias towards tags in preferred categories)
    # Build biased alpha for fine tags
    alpha_fine = np.ones(config.C_FINE) * 0.1  # Base small value
    
    # Boost tags in preferred categories
    for cat_idx, cat_weight in enumerate(cat_pref):
        if cat_weight > 0.1:  # Only boost significant categories
            for tag_idx in config.CATEGORY_TO_TAGS[cat_idx]:
                alpha_fine[tag_idx] += cat_weight * config.DIRICHLET_ALPHA_FINE * 5
    
    fine_pref_raw = rng.dirichlet(alpha_fine)
    fine_pref = fine_pref_raw / fine_pref_raw.sum()
    
    # 4. Vibe / personality profile (choose 3-10 core traits)
    num_vibe_tags = rng.integers(3, 11)
    selected_vibe_indices = rng.choice(config.C_VIBE, size=num_vibe_tags, replace=False)
    
    vibe_weights = rng.exponential(scale=1.0, size=num_vibe_tags)
    vibe_pref = np.zeros(config.C_VIBE)
    vibe_pref[selected_vibe_indices] = vibe_weights
    vibe_pref = vibe_pref / vibe_pref.sum()
    
    # 5. Behavior statistics (sample from configured ranges)
    avg_sessions_per_week = float(rng.uniform(*config.SESSIONS_PER_WEEK_RANGE))
    avg_views_per_session = float(rng.uniform(*config.VIEWS_PER_SESSION_RANGE))
    avg_likes_per_session = float(rng.uniform(*config.LIKES_PER_SESSION_RANGE))
    avg_saves_per_session = float(rng.uniform(*config.SAVES_PER_SESSION_RANGE))
    avg_attends_per_month = float(rng.uniform(*config.ATTENDS_PER_MONTH_RANGE))
    
    # 6. Location behavior (concentrate on 1-3 neighborhoods)
    num_frequent_neighborhoods = rng.integers(1, 4)
    frequent_neighborhoods = rng.choice(
        config.N_NEIGHBORHOODS_PER_CITY,
        size=num_frequent_neighborhoods,
        replace=False
    )
    
    # Dirichlet over frequent neighborhoods
    area_weights = rng.dirichlet([config.DIRICHLET_ALPHA_AREA] * num_frequent_neighborhoods)
    area_freqs: Dict[int, float] = {}
    for neigh_id, weight in zip(frequent_neighborhoods, area_weights):
        area_freqs[int(neigh_id)] = float(weight)
    
    return UserSchema(
        user_id=user_id,
        home_city_id=int(home_city_id),
        home_neighborhood_id=int(home_neighborhood_id),
        cat_pref=cat_pref.tolist(),
        fine_pref=fine_pref.tolist(),
        vibe_pref=vibe_pref.tolist(),
        area_freqs=area_freqs,
        avg_sessions_per_week=avg_sessions_per_week,
        avg_views_per_session=avg_views_per_session,
        avg_likes_per_session=avg_likes_per_session,
        avg_saves_per_session=avg_saves_per_session,
        avg_attends_per_month=avg_attends_per_month
    )


def generate_all_users(config: SyntheticConfig) -> List[UserSchema]:
    """
    Generate all synthetic users.
    
    Args:
        config: Generator configuration
    
    Returns:
        List of UserSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 1)  # Different seed from places
    
    users = []
    print(f"Generating {config.N_USERS} users...")
    
    for user_id in range(config.N_USERS):
        if (user_id + 1) % 1000 == 0:
            print(f"  Generated {user_id + 1}/{config.N_USERS} users")
        
        user = generate_user(user_id, config, rng)
        users.append(user)
    
    print(f"✅ Generated {len(users)} users")
    return users
```

### 7.4 Interaction Generation

**File**: `recsys/synthetic/generate_interactions.py`

```python
"""
Generate synthetic user-place interactions.
"""

import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from recsys.data.schemas import UserSchema, PlaceSchema, InteractionSchema, compute_implicit_rating
from recsys.synthetic.generator_config import SyntheticConfig


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_preference_score(
    user: UserSchema,
    place: PlaceSchema,
    config: SyntheticConfig
) -> float:
    """
    Compute preference score for user-place pair.
    
    Formula:
    score = w1 * sim_interest + w2 * sim_cat + w3 * loc_factor + w4 * popularity + noise
    """
    # Interest similarity (cosine of fine prefs)
    sim_interest = cosine_similarity(
        np.array(user.fine_pref),
        np.array(place.fine_tag_vector)
    )
    
    # Category alignment (dot product)
    sim_cat = np.dot(
        np.array(user.cat_pref),
        np.array(place.category_one_hot)
    )
    
    # Location factor (higher if place in frequent neighborhoods)
    loc_factor = user.area_freqs.get(place.neighborhood_id, 0.0)
    
    # Popularity factor (log-normalize)
    popularity = np.log1p(place.base_popularity) / 10.0
    
    # Combine with weights
    score = (
        config.W_INTEREST * sim_interest +
        config.W_CATEGORY * sim_cat +
        config.W_LOCATION * loc_factor +
        config.W_POPULARITY * popularity
    )
    
    return max(0.0, score)  # Ensure non-negative


def sample_interactions_for_user(
    user: UserSchema,
    places: List[PlaceSchema],
    config: SyntheticConfig,
    rng: np.random.Generator,
    start_date: datetime
) -> List[InteractionSchema]:
    """
    Sample interactions for a single user.
    """
    # 1. Determine number of interactions
    n_interactions_raw = int(
        user.avg_sessions_per_week *
        config.TIME_HORIZON_WEEKS *
        user.avg_views_per_session / 10.0  # Scale down
    )
    n_interactions = np.clip(
        n_interactions_raw,
        *config.INTERACTIONS_PER_USER_RANGE
    )
    
    # 2. Get candidate places (same city, occasionally others)
    same_city_places = [p for p in places if p.city_id == user.home_city_id]
    
    if len(same_city_places) < 20:
        # If not enough places in city, use all places
        candidate_places = places
    else:
        # 90% same city, 10% any city (travel)
        if rng.random() < 0.9:
            candidate_places = same_city_places
        else:
            candidate_places = places
    
    if len(candidate_places) == 0:
        return []
    
    # 3. Compute preference scores for all candidates
    scores = np.array([
        compute_preference_score(user, place, config)
        for place in candidate_places
    ])
    
    # Add noise
    scores = scores + rng.normal(0, 0.1, size=len(scores))
    scores = np.maximum(scores, 0.0)
    
    # 4. Convert to probabilities via softmax
    scores_exp = np.exp(scores - scores.max())  # Numerical stability
    probs = scores_exp / scores_exp.sum()
    
    # 5. Sample places according to probabilities (with replacement)
    sampled_indices = rng.choice(
        len(candidate_places),
        size=n_interactions,
        replace=True,
        p=probs
    )
    
    # 6. Generate interaction details for each sampled place
    interactions = []
    
    for idx in sampled_indices:
        place = candidate_places[idx]
        score = scores[idx]
        
        # Dwell time (higher score → more time)
        base_dwell = 30.0  # seconds
        dwell_time = base_dwell + score * 200.0 + rng.exponential(50.0)
        dwell_time = max(5.0, min(600.0, dwell_time))  # Cap at 10 minutes
        
        # Actions (higher score → more actions)
        action_prob = min(0.95, score * 2.0)
        
        num_likes = int(rng.random() < action_prob * 0.7)
        num_saves = int(rng.random() < action_prob * 0.4)
        num_shares = int(rng.random() < action_prob * 0.2)
        attended = bool(rng.random() < action_prob * 0.1)  # Attending is rare
        
        # Implicit rating
        implicit_rating = compute_implicit_rating(
            dwell_time, num_likes, num_saves, num_shares, attended
        )
        
        # Timestamp (random within time horizon)
        days_offset = rng.integers(0, config.TIME_HORIZON_WEEKS * 7)
        hours_offset = rng.integers(0, 24)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)
        
        # Time buckets
        hour = timestamp.hour
        if hour < 6:
            time_of_day_bucket = 3  # night
        elif hour < 12:
            time_of_day_bucket = 0  # morning
        elif hour < 18:
            time_of_day_bucket = 1  # afternoon
        else:
            time_of_day_bucket = 2  # evening
        
        day_of_week_bucket = 0 if timestamp.weekday() < 5 else 1  # weekday/weekend
        
        interaction = InteractionSchema(
            user_id=user.user_id,
            place_id=place.place_id,
            dwell_time=float(dwell_time),
            num_likes=num_likes,
            num_saves=num_saves,
            num_shares=num_shares,
            attended=attended,
            implicit_rating=float(implicit_rating),
            timestamp=timestamp,
            time_of_day_bucket=time_of_day_bucket,
            day_of_week_bucket=day_of_week_bucket
        )
        
        interactions.append(interaction)
    
    return interactions


def generate_all_interactions(
    users: List[UserSchema],
    places: List[PlaceSchema],
    config: SyntheticConfig,
    start_date: datetime = None
) -> List[InteractionSchema]:
    """
    Generate all user-place interactions.
    
    Args:
        users: List of users
        places: List of places
        config: Generator configuration
        start_date: Start date for timestamps
    
    Returns:
        List of InteractionSchema
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    rng = np.random.default_rng(config.RANDOM_SEED + 2)
    
    all_interactions = []
    print(f"Generating interactions for {len(users)} users...")
    
    for i, user in enumerate(users):
        if (i + 1) % 1000 == 0:
            print(f"  Generated interactions for {i + 1}/{len(users)} users")
        
        user_interactions = sample_interactions_for_user(
            user, places, config, rng, start_date
        )
        all_interactions.extend(user_interactions)
    
    print(f"✅ Generated {len(all_interactions)} interactions")
    return all_interactions
```

### 7.5 User-User Edge and Friend Label Generation

**File**: `recsys/synthetic/generate_user_user_edges.py`

```python
"""
Generate synthetic user-user edges and friend labels.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from recsys.data.schemas import UserSchema, InteractionSchema, UserUserEdgeSchema, FriendLabelSchema
from recsys.synthetic.generator_config import SyntheticConfig


def cosine_similarity_users(user_a: UserSchema, user_b: UserSchema) -> float:
    """Compute cosine similarity between two users based on fine preferences."""
    fine_a = np.array(user_a.fine_pref)
    fine_b = np.array(user_b.fine_pref)
    
    dot = np.dot(fine_a, fine_b)
    norm_a = np.linalg.norm(fine_a)
    norm_b = np.linalg.norm(fine_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))


def compute_neighborhood_overlap(user_a: UserSchema, user_b: UserSchema) -> float:
    """
    Compute overlap in area_freqs (Jaccard-like similarity).
    """
    neighborhoods_a = set(user_a.area_freqs.keys())
    neighborhoods_b = set(user_b.area_freqs.keys())
    
    if len(neighborhoods_a) == 0 or len(neighborhoods_b) == 0:
        return 0.0
    
    intersection = neighborhoods_a & neighborhoods_b
    union = neighborhoods_a | neighborhoods_b
    
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def build_co_attendance_map(interactions: List[InteractionSchema]) -> Dict[Tuple[int, int], int]:
    """
    Build map of co-attendance counts.
    
    Returns:
        Dict[(user_u, user_v)] -> count (where user_u < user_v)
    """
    # Group interactions by place and time window (day)
    place_day_users: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    
    for interaction in interactions:
        if not interaction.attended:
            continue
        
        place_id = interaction.place_id
        day = interaction.timestamp.date()
        place_day_users[(place_id, day)].add(interaction.user_id)
    
    # Count co-attendances
    co_attendance: Dict[Tuple[int, int], int] = defaultdict(int)
    
    for users_at_place in place_day_users.values():
        users_list = list(users_at_place)
        # Create pairs
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                user_u = min(users_list[i], users_list[j])
                user_v = max(users_list[i], users_list[j])
                co_attendance[(user_u, user_v)] += 1
    
    return dict(co_attendance)


def generate_social_edges(
    users: List[UserSchema],
    interactions: List[InteractionSchema],
    config: SyntheticConfig
) -> List[UserUserEdgeSchema]:
    """
    Generate user-user social edges.
    
    Args:
        users: List of users
        interactions: List of interactions (for co-attendance)
        config: Generator configuration
    
    Returns:
        List of UserUserEdgeSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 3)
    
    print("Building co-attendance map...")
    co_attendance_map = build_co_attendance_map(interactions)
    
    print(f"Finding similar users for {len(users)} users...")
    
    # Build user index for fast lookup
    user_dict = {user.user_id: user for user in users}
    
    edges = []
    processed_pairs = set()
    
    for i, user_a in enumerate(users):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(users)} users")
        
        # Find similar users in same city
        candidates = [
            user_b for user_b in users
            if user_b.user_id != user_a.user_id and
            user_b.home_city_id == user_a.home_city_id
        ]
        
        if len(candidates) == 0:
            continue
        
        # Compute similarities
        similarities = [
            (user_b, cosine_similarity_users(user_a, user_b))
            for user_b in candidates
        ]
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        
        # Take top-K similar users
        top_similar = similarities[:config.K_SIMILAR_USERS]
        
        for user_b, sim_score in top_similar:
            if sim_score < config.SIMILARITY_THRESHOLD:
                continue
            
            # Ensure consistent ordering (smaller ID first)
            user_u_id = min(user_a.user_id, user_b.user_id)
            user_v_id = max(user_a.user_id, user_b.user_id)
            
            # Skip if already processed
            if (user_u_id, user_v_id) in processed_pairs:
                continue
            
            processed_pairs.add((user_u_id, user_v_id))
            
            user_u = user_dict[user_u_id]
            user_v = user_dict[user_v_id]
            
            # Compute edge features
            interest_overlap_score = sim_score
            co_attendance_count = co_attendance_map.get((user_u_id, user_v_id), 0)
            same_neighborhood_freq = compute_neighborhood_overlap(user_u, user_v)
            
            edge = UserUserEdgeSchema(
                user_u=user_u_id,
                user_v=user_v_id,
                interest_overlap_score=float(interest_overlap_score),
                co_attendance_count=int(co_attendance_count),
                same_neighborhood_freq=float(same_neighborhood_freq)
            )
            
            edges.append(edge)
    
    print(f"✅ Generated {len(edges)} social edges")
    return edges


def generate_friend_labels(
    edges: List[UserUserEdgeSchema],
    users: List[UserSchema],
    config: SyntheticConfig
) -> List[FriendLabelSchema]:
    """
    Generate friend compatibility labels from edges.
    
    Args:
        edges: Social edges
        users: List of users
        config: Generator configuration
    
    Returns:
        List of FriendLabelSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 4)
    
    print("Generating friend labels...")
    
    labels = []
    
    # Positive labels from edges
    for edge in edges:
        # Compatibility: high similarity or co-attendance
        is_compatible = (
            edge.interest_overlap_score >= 0.5 or
            edge.co_attendance_count >= 2
        )
        
        label_compat = 1 if is_compatible else 0
        
        # Attendance: probability based on similarity and co-attendance
        attend_prob = min(
            0.9,
            edge.interest_overlap_score * 0.5 +
            min(edge.co_attendance_count / 10.0, 0.4) +
            edge.same_neighborhood_freq * 0.1
        )
        
        label_attend = 1 if rng.random() < attend_prob else 0
        
        labels.append(FriendLabelSchema(
            user_u=edge.user_u,
            user_v=edge.user_v,
            label_compat=label_compat,
            label_attend=label_attend
        ))
    
    # Negative labels (sample random pairs with no edge)
    # Sample ~20% of positive labels as negatives
    num_negatives = len(labels) // 5
    
    user_ids = [u.user_id for u in users]
    edge_set = {(edge.user_u, edge.user_v) for edge in edges}
    
    attempts = 0
    max_attempts = num_negatives * 10
    
    while len(labels) - len(edges) < num_negatives and attempts < max_attempts:
        attempts += 1
        
        # Sample two random users
        user_u_id, user_v_id = rng.choice(user_ids, size=2, replace=False)
        user_u_id, user_v_id = min(user_u_id, user_v_id), max(user_u_id, user_v_id)
        
        # Skip if edge exists
        if (user_u_id, user_v_id) in edge_set:
            continue
        
        # Skip if already added as negative
        if any(l.user_u == user_u_id and l.user_v == user_v_id for l in labels[len(edges):]):
            continue
        
        # Negative label
        labels.append(FriendLabelSchema(
            user_u=user_u_id,
            user_v=user_v_id,
            label_compat=0,
            label_attend=0
        ))
    
    print(f"✅ Generated {len(labels)} friend labels ({len(edges)} positive, {len(labels) - len(edges)} negative)")
    return labels
```

### 7.6 Orchestration Script

**File**: `scripts/run_synthetic_generation.py`

```python
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
    data_dicts = [vars(item) for item in data_list]
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
```

**Usage**:

```bash
# Generate with defaults (10k users, 10k places)
python scripts/run_synthetic_generation.py --output_dir data/

# Generate smaller dataset for testing
python scripts/run_synthetic_generation.py \
    --output_dir data/test/ \
    --n_users 1000 \
    --n_places 1000

# Generate larger dataset
python scripts/run_synthetic_generation.py \
    --output_dir data/full/ \
    --n_users 50000 \
    --n_places 20000
```

**Expected output**:

```
================================================================================
SYNTHETIC DATA GENERATION
================================================================================

Configuration:
  Users: 10000
  Places: 10000
  Cities: 8
  Random seed: 42

Step 1/5: Generating places...
Generating 10000 places...
  Generated 1000/10000 places
  Generated 2000/10000 places
  ...
✅ Generated 10000 places
  Saved to data/places.parquet

Step 2/5: Generating users...
Generating 10000 users...
  Generated 1000/10000 users
  ...
✅ Generated 10000 users
  Saved to data/users.parquet

Step 3/5: Generating interactions...
Generating interactions for 10000 users...
  Generated interactions for 1000/10000 users
  ...
✅ Generated 2847293 interactions
  Saved to data/interactions.parquet

Step 4/5: Generating social edges...
Building co-attendance map...
Finding similar users for 10000 users...
  Processed 1000/10000 users
  ...
✅ Generated 156432 social edges
  Saved to data/user_user_edges.parquet

Step 5/5: Generating friend labels...
Generating friend labels...
✅ Generated 187718 friend labels (156432 positive, 31286 negative)
  Saved to data/friend_labels.parquet

================================================================================
GENERATION COMPLETE
================================================================================

Generated data:
  Users: 10000
  Places: 10000
  Interactions: 2847293
  Social edges: 156432
  Friend labels: 187718

Files saved to: /path/to/data

Next steps:
  1. Run scripts/run_build_features.py to build the graph
  2. Run scripts/run_train_gnn.py to train the model
```

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


