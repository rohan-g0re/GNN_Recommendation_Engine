## 1. Scope and Objectives

This document provides a **low-level design (LLD)** for the **GNN-based recommendation engine backend** that powers:

- **Place recommendations** (spatial analysis).
- **People recommendations** (social compatibility + incentive to attend).

The design assumes:

- **Python** as the implementation language.
- **PyTorch + PyTorch Geometric** for the GNN.
- **FastAPI** for HTTP APIs.
- **Approximate nearest neighbor (ANN)** indices for fast retrieval.
- **Synthetic data** for initial development and testing.

The goal is to make implementation as straightforward as possible by specifying:

- **Directory structure**.
- **Modules, classes, and their responsibilities**.
- **Data schemas and feature flows**.
- **Training pipeline**.
- **Serving and explanation logic**.

---

## 2. High-Level Architecture

At a high level, the system consists of:

- **Offline components**:
  - **Synthetic data generator**: creates users, places, interactions, and social edges.
  - **Feature and graph builder**: converts raw/synthetic data into feature tensors and a heterogeneous graph.
  - **GNN training pipeline**: trains user/place encoders, GNN backbone, and task heads.
  - **Embedding exporter and ANN index builder**: exports embeddings and builds indices.

- **Online components**:
  - **Recommendation service** (FastAPI):
    - `/recommend/places` endpoint.
    - `/recommend/people` endpoint.
  - **ANN managers**:
    - For user and place embeddings.
  - **Explanation service**:
    - Converts feature overlaps into human-readable explanations.

---

## 3. Project Structure

Proposed directory layout:

```text
recsys/
  config/
    __init__.py
    settings.py
  data/
    __init__.py
    schemas.py
    repositories.py
  synthetic/
    __init__.py
    generator_config.py
    generate_users.py
    generate_places.py
    generate_interactions.py
    generate_user_user_edges.py
  features/
    __init__.py
    user_features.py
    place_features.py
    interaction_features.py
    graph_builder.py
  ml/
    __init__.py
    models/
      __init__.py
      encoders.py
      backbone.py
      heads.py
      losses.py
    training/
      __init__.py
      datasets.py
      sampler.py
      train_loop.py
      eval_metrics.py
  serving/
    __init__.py
    ann_index.py
    recommender_core.py
    explanations.py
    api_schemas.py
    api_main.py
  scripts/
    run_synthetic_generation.py
    run_build_features.py
    run_train_gnn.py
    run_export_embeddings.py
    run_build_indices.py
```

You can adapt naming, but this structure separates **data**, **ML**, and **serving** clearly.

---

## 4. Configuration and Settings (`config/`)

### 4.1 `config/constants.py`

**CRITICAL**: These constants MUST match exactly between training and serving.

```python
# File: recsys/config/constants.py

# Scale parameters
N_USERS = 10_000
N_PLACES = 10_000
N_CITIES = 8
N_NEIGHBORHOODS_PER_CITY = 15

# Feature dimensions (MUST MATCH between all modules)
C_COARSE = 6  # Number of coarse categories
C_FINE = 100  # Number of fine-grained tags
C_VIBE = 30   # Number of vibe/personality tags

# Computed feature dimensions
D_USER_RAW = 148  # User feature vector dimension
D_PLACE_RAW = 114  # Place feature vector dimension
D_EDGE_UP = 12    # User-place edge features
D_EDGE_UU = 3     # User-user edge features

MAX_NEIGHBORHOODS_PER_USER = 5  # Fixed-size for area_freqs

# Coarse categories (0-indexed)
COARSE_CATEGORIES = [
    "entertainment",  # 0
    "sports",         # 1
    "clubs",          # 2
    "dining",         # 3
    "outdoors",       # 4
    "culture"         # 5
]

# Fine tags (0-indexed, example subset shown)
FINE_TAGS = [
    "fishing", "bouldering", "techno", "live_music", "board_games",
    "rooftop", "brunch", "karaoke", "jazz", "trivia",
    # ... 90 more to total 100
]

# Vibe tags (0-indexed, example subset shown)
VIBE_TAGS = [
    "introvert", "extrovert", "party", "fitness", "artsy",
    "night_owl", "early_bird", "foodie", "adventurous", "chill",
    # ... 20 more to total 30
]

# Time slots
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
```

### 4.2 `config/model_config.py`

**CRITICAL**: Model configuration shared between training and serving.

```python
# File: recsys/config/model_config.py

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Hyperparameters for GNN model.
    MUST be consistent between training and serving.
    """
    # Embedding dimension
    D_MODEL: int = 128  # Output dimension of encoders and GNN layers
    
    # Encoder architecture
    CITY_EMBED_DIM: int = 16
    NEIGHBORHOOD_EMBED_DIM: int = 32
    PRICE_EMBED_DIM: int = 8
    TIME_SLOT_EMBED_DIM: int = 8
    
    # GNN backbone
    NUM_GNN_LAYERS: int = 2  # Number of message passing layers
    GNN_HIDDEN_DIM: int = 128  # Hidden dimension in conv layers
    GNN_DROPOUT: float = 0.1
    GNN_AGGR: str = 'mean'  # Aggregation: 'mean', 'sum', or 'attention'
    
    # Task heads
    PLACE_HEAD_HIDDEN: list = None  # [256, 128]
    FRIEND_HEAD_HIDDEN: list = None  # [256, 128]
    HEAD_DROPOUT: float = 0.2
    
    # Context vector dimensions
    D_CTX_PLACE: int = 16  # Dimension of place context vector
    D_CTX_FRIEND: int = 16  # Dimension of friend context vector
    
    # Training
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    BATCH_SIZE_PLACE: int = 512
    BATCH_SIZE_FRIEND: int = 512
    
    # Loss weights
    LAMBDA_PLACE: float = 1.0
    LAMBDA_FRIEND: float = 0.5
    LAMBDA_ATTEND: float = 0.3
    
    def __post_init__(self):
        if self.PLACE_HEAD_HIDDEN is None:
            self.PLACE_HEAD_HIDDEN = [256, 128]
        if self.FRIEND_HEAD_HIDDEN is None:
            self.FRIEND_HEAD_HIDDEN = [256, 128]
```

### 4.3 `config/settings.py`

File paths and runtime settings:

```python
# File: recsys/config/settings.py

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data files
USERS_FILE = DATA_DIR / "users.parquet"
PLACES_FILE = DATA_DIR / "places.parquet"
INTERACTIONS_FILE = DATA_DIR / "interactions.parquet"
USER_USER_EDGES_FILE = DATA_DIR / "user_user_edges.parquet"
FRIEND_LABELS_FILE = DATA_DIR / "friend_labels.parquet"

# Graph and mappings
GRAPH_FILE = DATA_DIR / "hetero_graph.pt"
USER_ID_MAPPINGS_FILE = DATA_DIR / "user_id_mappings.pkl"
PLACE_ID_MAPPINGS_FILE = DATA_DIR / "place_id_mappings.pkl"

# Model checkpoints
MODEL_CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FINAL_MODEL_PATH = MODEL_DIR / "final_model.pt"

# Embeddings
EMBEDDING_DIR = DATA_DIR / "embeddings"
USER_EMBEDDINGS_FILE = EMBEDDING_DIR / "user_embeddings.parquet"
PLACE_EMBEDDINGS_FILE = EMBEDDING_DIR / "place_embeddings.parquet"

# ANN indices
INDEX_DIR = DATA_DIR / "indices"

# Serving parameters
TOP_M_CANDIDATES = 200  # ANN retrieval count
TOP_K_RESULTS = 10      # Final recommendations
```

---

## 5. Data Layer (`data/`)

### 5.1 Schemas (`data/schemas.py`)

**CRITICAL**: All dimensions and field names must match exactly.

```python
# File: recsys/data/schemas.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

@dataclass
class UserSchema:
    """
    User data schema.
    Raw feature dimension: 148
    """
    # Identity
    user_id: int  # 0 to N_USERS-1
    
    # Location (categorical, will be embedded)
    home_city_id: int  # 0 to N_CITIES-1
    home_neighborhood_id: int  # 0 to N_NEIGHBORHOODS_PER_CITY-1
    
    # Preference vectors (pre-normalized, sum to 1.0)
    cat_pref: List[float]  # Length C_COARSE=6
    fine_pref: List[float]  # Length C_FINE=100
    vibe_pref: List[float]  # Length C_VIBE=30
    
    # Location behavior (will be converted to fixed-size vector)
    area_freqs: Dict[int, float]  # neighborhood_id -> weight, sums to 1.0
    
    # Behavioral statistics (continuous, normalized to [0,1])
    avg_sessions_per_week: float
    avg_views_per_session: float
    avg_likes_per_session: float
    avg_saves_per_session: float
    avg_attends_per_month: float
    
    # Optional demographics
    age_group: Optional[int] = None
    gender: Optional[int] = None


@dataclass
class PlaceSchema:
    """
    Place data schema.
    Raw feature dimension: 114
    """
    # Identity
    place_id: int  # 0 to N_PLACES-1
    
    # Location
    city_id: int  # 0 to N_CITIES-1
    neighborhood_id: int  # 0 to N_NEIGHBORHOODS_PER_CITY-1
    
    # Categories (can be multi-label)
    category_ids: List[int]  # Indices into COARSE_CATEGORIES
    category_one_hot: List[float]  # Length C_COARSE=6, binary or weighted
    
    # Fine-grained tags (normalized, sum to 1.0)
    fine_tag_vector: List[float]  # Length C_FINE=100
    
    # Operational attributes
    price_band: int  # 0-4
    typical_time_slot: int  # 0-5
    
    # Popularity metrics
    base_popularity: float  # Raw popularity score
    avg_daily_visits: float  # Derived
    conversion_rate: float  # 0-1
    novelty_score: float  # 0-1


@dataclass
class InteractionSchema:
    """
    User-place interaction.
    Edge feature dimension: 12
    """
    # Foreign keys
    user_id: int
    place_id: int
    
    # Engagement metrics
    dwell_time: float  # Seconds
    num_likes: int
    num_saves: int
    num_shares: int
    attended: bool
    
    # Derived implicit rating (1.0-5.0)
    implicit_rating: float
    
    # Temporal context
    timestamp: datetime
    time_of_day_bucket: int  # 0-3
    day_of_week_bucket: int  # 0-1


@dataclass
class UserUserEdgeSchema:
    """
    User-user social edge.
    Edge feature dimension: 3
    """
    user_u: int  # Smaller ID
    user_v: int  # Larger ID
    
    # Edge strength features
    interest_overlap_score: float  # 0-1, cosine similarity
    co_attendance_count: int  # Number of co-attended places
    same_neighborhood_freq: float  # 0-1, overlap in area_freqs


@dataclass
class FriendLabelSchema:
    """
    Supervision labels for friend compatibility.
    """
    user_u: int
    user_v: int
    label_compat: int  # 0 or 1
    label_attend: int  # 0 or 1
```

### 5.1.1 Implicit Rating Computation

**CRITICAL**: This formula must be used consistently for synthetic data generation and preprocessing.

```python
# File: recsys/data/schemas.py (continued)

def compute_implicit_rating(
    dwell_time: float,
    num_likes: int,
    num_saves: int,
    num_shares: int,
    attended: bool
) -> float:
    """
    Convert engagement signals to 1-5 rating scale.
    
    Formula:
    - Base: 1.0
    - +min(dwell_time / 150, 2.0) for dwell
    - +min(num_likes * 0.5, 1.5) for likes
    - +min(num_saves * 1.0, 2.0) for saves
    - +min(num_shares * 0.5, 1.0) for shares
    - +2.0 if attended
    - Cap at 5.0
    """
    score = 1.0
    score += min(dwell_time / 150.0, 2.0)
    score += min(num_likes * 0.5, 1.5)
    score += min(num_saves * 1.0, 2.0)
    score += min(num_shares * 0.5, 1.0)
    if attended:
        score += 2.0
    return min(score, 5.0)
```

### 5.1.2 Data Validation

```python
# File: recsys/data/validators.py

from recsys.config.constants import *

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

### 5.2 Repositories (`data/repositories.py`)

Abstract data access behind repository classes:

- **`UserRepository`**
  - Backed by CSV/Parquet or DB.
  - Methods:
    - `get_user(user_id: int) -> UserSchema`
    - `get_users_by_city(city_id: int) -> List[UserSchema]`
    - `get_all_users() -> Iterable[UserSchema]`

- **`PlaceRepository`**
  - Methods:
    - `get_place(place_id: int) -> PlaceSchema`
    - `get_places_by_city(city_id: int) -> List[PlaceSchema]`
    - `get_all_places() -> Iterable[PlaceSchema]`

- **`InteractionRepository`**
  - Methods:
    - `get_interactions_for_user(user_id: int) -> List[InteractionSchema]`
    - `get_interactions_for_place(place_id: int) -> List[InteractionSchema]`
    - `get_all_interactions() -> Iterable[InteractionSchema]`

- **`UserUserRepository`**
  - Methods:
    - `get_edges_for_user(user_id: int) -> List[UserUserEdgeSchema]`
    - `get_all_edges() -> Iterable[UserUserEdgeSchema]`

- **`EmbeddingRepository`**
  - For online serving.
  - Methods:
    - `get_user_embedding(user_id: int) -> np.ndarray`
    - `get_place_embedding(place_id: int) -> np.ndarray`
    - `set_user_embedding(user_id: int, embedding: np.ndarray)`
    - `set_place_embedding(place_id: int, embedding: np.ndarray)`

Implementation:

- For the assignment, these can be:
  - Simple CSV/parquet readers using `pandas` or `pyarrow`.
  - In-memory dictionaries loaded at startup for serving.

---

## 6. Synthetic Data Generation (`synthetic/`)

This layer creates the synthetic world described in the GNN plan.

### 6.1 `synthetic/generator_config.py`

- Define:
  - `N_USERS`, `N_PLACES`, `N_CITIES`.
  - `COARSE_CATEGORIES` (list of names or IDs).
  - `FINE_TAGS` (list of strings).
  - `VIBE_TAGS` (list of strings).
  - Distribution parameters for Dirichlet and log-normal distributions.

### 6.2 `synthetic/generate_users.py`

Key functions:

- **`generate_user(user_id: int, config) -> UserSchema`**
  - Step 1: Sample `home_city_id` and `home_neighborhood_id`.
  - Step 2: Sample `cat_pref` via `Dirichlet(alpha_cat)`.
  - Step 3: Sample `fine_pref` via category-conditioned Dirichlet.
  - Step 4: Sample a subset of `VIBE_TAGS` and create `vibe_pref`.
  - Step 5: Sample behavior stats.
  - Step 6: Sample `area_freqs` over neighborhoods in the city.

- **`generate_all_users(config) -> List[UserSchema]`**
  - Loop over `user_id` from `0` to `N_USERS - 1`.
  - Call `generate_user`.
  - Save as `users.parquet` or similar.

### 6.3 `synthetic/generate_places.py`

Key functions:

- **`generate_place(place_id: int, config) -> PlaceSchema`**
  - Step 1: Sample `city_id` and `neighborhood_id`.
  - Step 2: Sample 1–2 coarse categories.
  - Step 3: Sample fine tags consistent with those categories and build `fine_tag_vector`.
  - Step 4: Sample `price_band`, `typical_time_slot`.
  - Step 5: Sample `base_popularity` and derive `avg_daily_visits`, `conversion_rate`, `novelty_score`.

- **`generate_all_places(config) -> List[PlaceSchema]`**
  - Loop over `place_id`.
  - Save as `places.parquet`.

### 6.4 `synthetic/generate_interactions.py`

Key functions:

- **`compute_preference_score(user: UserSchema, place: PlaceSchema) -> float`**
  - Compute:
    - Cosine similarity of `user_fine_pref` and `place_fine_tags`.
    - Dot product of `user_cat_pref` and `place.category_one_hot`.
    - Location factor from `area_freqs` and place neighborhood.
    - Popularity factor from `base_popularity`.
    - Combine with tunable weights and noise.

- **`sample_interactions_for_user(user: UserSchema, places_in_city: List[PlaceSchema]) -> List[InteractionSchema]`**
  - Step 1: Determine `N_interactions`.
  - Step 2: Compute `score_up` for all candidate places.
  - Step 3: Convert to probabilities via softmax.
  - Step 4: Sample places according to probabilities.
  - Step 5: For each `(user, place)`:
    - Sample dwell time.
    - Sample actions (like/save/attend) based on score.
    - Derive `implicit_rating`.
    - Sample timestamp and convert to `time_of_day_bucket`, `day_of_week_bucket`.

- **`generate_all_interactions(config, users, places) -> List[InteractionSchema]`**
  - For each user:
    - Filter places by user city.
    - Call `sample_interactions_for_user`.
  - Save as `interactions.parquet`.

### 6.5 `synthetic/generate_user_user_edges.py`

Key functions:

- **`build_user_similarity_index(users: List[UserSchema])`**
  - Build a simple ANN index on `user_fine_pref` (and optionally `vibe_pref`).

- **`generate_social_edges(users, interactions) -> List[UserUserEdgeSchema]`**
  - For each user:
    - Find top-k similar users using the similarity index.
    - Compute:
      - `interest_overlap_score`.
      - `co_attendance_count` from overlapping visits (based on `interactions`).
      - `same_neighborhood_freq` using `area_freqs`.
    - Create `UserUserEdgeSchema` if combined score > threshold.

- **`generate_friend_labels(edges) -> List[FriendLabelSchema]`**
  - Positive labels:
    - Edges with high `interest_overlap_score` and/or `co_attendance_count`.
  - Negative labels:
    - Random pairs with low similarity and no co-attendance.
  - For attendance:
    - Simulate acceptance probability as an increasing function of similarity and co-attendance.
    - Sample 0/1 `label_attend`.

### 6.6 `scripts/run_synthetic_generation.py`

Orchestration script:

- Load `config`.
- Call:
  - `generate_all_users`.
  - `generate_all_places`.
  - `generate_all_interactions`.
  - `generate_social_edges` and `generate_friend_labels`.
- Save datasets to `DATA_DIR`.

---

## 7. Feature Extraction and Graph Building (`features/`)

### 7.1 User Features (`features/user_features.py`)

Functions:

- **`encode_user_features(user: UserSchema, config) -> np.ndarray`**
  - Take:
    - `cat_pref`, `fine_pref`, `vibe_pref` as float vectors.
    - `home_city_id`, `home_neighborhood_id` as categorical indices.
    - Behavior stats and `area_freqs`.
  - Produce:
    - A flattened numeric vector for the user encoder input.

- **`build_user_feature_matrix(users: List[UserSchema]) -> np.ndarray`**
  - Build an `N_users x D_user_features` matrix.

### 7.2 Place Features (`features/place_features.py`)

Functions:

- **`encode_place_features(place: PlaceSchema, config) -> np.ndarray`**
  - Take:
    - `category_one_hot`, `fine_tag_vector`.
    - Encoded `city_id`, `neighborhood_id`.
    - `price_band`, `typical_time_slot`.
    - Popularity features.
  - Produce a flattened input vector.

- **`build_place_feature_matrix(places: List[PlaceSchema]) -> np.ndarray`**
  - Build an `N_places x D_place_features` matrix.

### 7.3 Interaction Features (`features/interaction_features.py`)

Functions:

- **`encode_interaction_features(interaction: InteractionSchema) -> np.ndarray`**
  - Build edge feature vector containing:
    - `implicit_rating`.
    - Normalized `dwell_time`.
    - `num_likes`, `num_saves`, `num_shares`.
    - `attended` as 0/1.
    - One-hot `time_of_day_bucket`.
    - One-hot `day_of_week_bucket`.

- **`build_user_place_edge_arrays(interactions: List[InteractionSchema], user_id_map, place_id_map) -> (edge_index, edge_attr)`**
  - Map `user_id` and `place_id` to contiguous indices.
  - Build:
    - `edge_index` as a 2 x E tensor.
    - `edge_attr` as E x D_edge_features tensor.

### 7.4 User–User Edge Features (`features/graph_builder.py` or separate file)

- **`build_user_user_edge_arrays(edges: List[UserUserEdgeSchema], user_id_map) -> (edge_index, edge_attr)`**
  - Map `user_u`, `user_v` to indices.
  - Edge features:
    - `interest_overlap_score`.
    - `co_attendance_count` (normalized).
    - `same_neighborhood_freq`.

### 7.5 Graph Builder (`features/graph_builder.py`)

Key function:

- **`build_hetero_graph(users, places, interactions, user_user_edges) -> HeteroData`**
  - Create mappings:
    - `user_id_to_index`, `index_to_user_id`.
    - `place_id_to_index`, `index_to_place_id`.
  - Build:
    - `data['user'].x` from user feature matrix.
    - `data['place'].x` from place feature matrix.
    - `data['user', 'interacts', 'place'].edge_index` and `edge_attr`.
    - `data['place', 'rev_interacts', 'user'].edge_index` (reverse relation).
    - `data['user', 'social', 'user'].edge_index` and `edge_attr`.
  - Serialize:
    - Save `HeteroData` object to `GRAPH_FILE`.
    - Save mapping dictionaries for later use (embeddings export, serving).

### 7.6 `scripts/run_build_features.py`

Orchestration script:

- Load generated synthetic datasets from `DATA_DIR`.
- Build feature matrices and graph.
- Save them for the training module.

---

## 8. GNN Model and Training (`ml/`)

### 8.1 Encoders (`ml/models/encoders.py`)

- **`UserEncoder(nn.Module)`**
  - Inputs:
    - `user_feature_tensor` of shape `(N_users, D_user_features)`.
  - Components:
    - Embedding layers for:
      - `home_city_id`.
      - `home_neighborhood_id`.
      - Possibly discrete buckets like `price_band`, `typical_time_slot` if re-used.
    - Linear layers to project concatenated numeric features.
  - Forward:
    - Embed categorical features.
    - Concatenate with numeric vectors (`cat_pref`, `fine_pref`, `vibe_pref`, behavior stats, projected `area_freqs`).
    - Apply MLP to map to `D_MODEL`:
      - Example architecture:
        - Linear(D_in, 2 * D_MODEL) → ReLU → Linear(2 * D_MODEL, D_MODEL).

- **`PlaceEncoder(nn.Module)`**
  - Similar pattern:
    - Embeddings for city, neighborhood, price band, time slot.
    - Concatenate with `category_one_hot`, `fine_tag_vector`, popularity metrics.
    - MLP to `D_MODEL`.

Both encoders can be applied ahead of the GNN, producing `x_user` and `x_place` tensors.

### 8.2 Backbone (`ml/models/backbone.py`)

- **`GraphRecBackbone(nn.Module)`**
  - Attributes:
    - `convs`: list of `HeteroConv` layers, length `NUM_GNN_LAYERS`.
  - Initialization:
    - For each layer:
      - Define relation-specific GNNs:
        - For `('user', 'social', 'user')`: e.g., GATConv or SAGEConv.
        - For `('user', 'interacts', 'place')` and `('place', 'rev_interacts', 'user')`: e.g., edge-aware SAGEConv.
  - Forward signature:
    - `forward(x_dict, edge_index_dict, edge_attr_dict) -> Dict[str, Tensor]`
  - Forward steps:
    - `x_dict` initially contains:
      - `{'user': x_user, 'place': x_place}` from encoders.
    - For each layer:
      - Apply `HeteroConv`:
        - Each relation uses its respective conv with `edge_index` and optionally `edge_attr`.
      - Optionally apply residual connections and normalization.
    - Return:
      - `z_dict = {'user': z_user, 'place': z_place}`.

### 8.3 Heads (`ml/models/heads.py`)

- **`PlaceHead(nn.Module)`**
  - Inputs:
    - `z_u_batch`: `(batch_size, D_MODEL)` user embeddings.
    - `z_p_batch`: `(batch_size, D_MODEL)` place embeddings.
    - `ctx_batch`: `(batch_size, D_ctx)` context features.
  - Construction:
    - Compute interaction features:
      - `z_mul = z_u_batch * z_p_batch`.
      - `z_diff = torch.abs(z_u_batch - z_p_batch)`.
    - Concatenate `[z_u_batch, z_p_batch, z_mul, z_diff, ctx_batch]`.
    - Pass through MLP to output scalar scores.

- **`FriendHead(nn.Module)`**
  - Inputs:
    - `z_u_batch`, `z_v_batch`, `ctx_batch`.
  - Construction:
    - Similar to `PlaceHead`, but with two output heads:
      - One MLP for `compat_score`.
      - One MLP followed by `sigmoid` for `attend_prob`.

### 8.4 Losses (`ml/models/losses.py`)

- **`bpr_loss(pos_scores, neg_scores)`**
  - Implements:

    \\[
    -\log \sigma(\text{pos} - \text{neg})
    \\]

- **`binary_cross_entropy_loss(pred, label)`**
  - Wrapper around PyTorch BCE/BCEWithLogits.

### 8.5 Training Datasets and Sampling (`ml/training/datasets.py`, `ml/training/sampler.py`)

- **`UserPlacePairDataset`**
  - Backed by:
    - User–place interactions for positives.
    - Pre-sampled negatives per user.
  - Each element:
    - `(user_index, pos_place_index, neg_place_index, ctx_features)`.

- **`UserUserPairDataset`**
  - Backed by:
    - Positive and negative friend labels.
  - Each element:
    - `(user_u_index, user_v_index, label_compat, label_attend, ctx_features)`.

- **Negative sampling utilities** in `sampler.py`:
  - For place task:
    - For each user, sample negative places from the same city and (optionally) same coarse category.
  - For friend task:
    - Sample negative users in the same city with low similarity.

### 8.6 Training Loop (`ml/training/train_loop.py`)

Core function:

- **`train_gnn(config, graph, user_place_dataset, user_user_dataset)`**
  - Initialize:
    - `UserEncoder`, `PlaceEncoder`.
    - `GraphRecBackbone`.
    - `PlaceHead`, `FriendHead`.
    - Optimizer (e.g., Adam).
  - For each epoch:
    - Sample batches from user–place dataset:
      - Obtain `user_index`, `pos_place_index`, `neg_place_index`.
      - Extract corresponding subgraph (optionally using neighbor sampling).
      - Forward pass:
        - Encoders → Backbone → embeddings `z_user`, `z_place`.
        - Use `PlaceHead` to compute scores for `pos` and `neg`.
      - Compute `L_place` via BPR or BCE.
    - Sample batches from user–user dataset:
      - Forward pass to get `z_u`, `z_v`.
      - Use `FriendHead` to compute compatibility and attendance.
      - Compute `L_compat` and `L_attend`.
    - Combine losses:
      - `L_total = lambda_place * L_place + lambda_friend * L_compat + lambda_attend * L_attend`.
    - Backpropagation and optimizer step.
    - Periodically evaluate on held-out validation data.
  - Save:
    - Best-performing encoder, backbone, and heads to `MODEL_CHECKPOINT_DIR`.

### 8.7 Evaluation Metrics (`ml/training/eval_metrics.py`)

- **Place recommendation metrics**:
  - Recall@K.
  - NDCG@K.
- **Friend recommendation metrics**:
  - ROC-AUC for compatibility.
  - Recall@K for predicting co-attendance.

---

## 9. Embedding Export and ANN Indexing (`scripts/`, `serving/ann_index.py`)

### 9.1 Embedding Export (`scripts/run_export_embeddings.py`)

Steps:

1. Load graph, mappings, and trained model (encoders + backbone).
2. Compute embeddings:
   - `z_user_all` for all users.
   - `z_place_all` for all places.
3. Map internal indices to IDs using stored mappings.
4. Save embeddings to disk:
   - `user_embeddings.parquet` with columns:
     - `user_id`, `embedding` (array).
   - `place_embeddings.parquet` with columns:
     - `place_id`, `embedding`.

### 9.2 ANN Index Abstraction (`serving/ann_index.py`)

Define a simple wrapper over an ANN library (e.g., Faiss):

- **`AnnIndex`**
  - Methods:
    - `build(vectors: np.ndarray, ids: List[int]) -> None`
    - `save(path: str) -> None`
    - `load(path: str) -> None`
    - `search(query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]`

- **`PlaceAnnIndexManager`**
  - Maintains a mapping:
    - `city_id -> AnnIndex` for places.

- **`UserAnnIndexManager`**
  - Maintains:
    - `city_id -> AnnIndex` for users.

### 9.3 Index Building Script (`scripts/run_build_indices.py`)

Steps:

1. Load user and place embeddings from `EMBEDDING_DIR`.
2. Group embeddings by city for:
   - Places: using `PlaceRepository` to map `place_id` to `city_id`.
   - Users: using `UserRepository` mapping.
3. For each city:
   - Build a place index and user index.
4. Save indexes to `INDEX_DIR`.

---

## 10. Serving Layer (`serving/`)

### 10.1 API Schemas (`serving/api_schemas.py`)

Use Pydantic models for request and response payloads.

- **Place recommendation request**

  - `PlaceRecommendationRequest`:
    - `user_id: int`
    - `city_id: Optional[int] = None`
    - `time_slot: Optional[int] = None`
    - `desired_categories: Optional[List[int]] = None`
    - `desired_tags: Optional[List[int]] = None`
    - `top_k: int = 10`

- **Place recommendation response**

  - `PlaceRecommendation`:
    - `place_id: int`
    - `score: float`
    - `explanations: List[str]`
  - `PlaceRecommendationResponse`:
    - `recommendations: List[PlaceRecommendation]`

- **People recommendation request**

  - `PeopleRecommendationRequest`:
    - `user_id: int`
    - `city_id: Optional[int] = None`
    - `target_place_id: Optional[int] = None`
    - `activity_tags: Optional[List[int]] = None`
    - `top_k: int = 10`

- **People recommendation response**

  - `PeopleRecommendation`:
    - `user_id: int`
    - `compat_score: float`
    - `attend_prob: float`
    - `explanations: List[str]`
  - `PeopleRecommendationResponse`:
    - `recommendations: List[PeopleRecommendation]`

### 10.2 Core Recommenders (`serving/recommender_core.py`)

- **`PlaceRecommender`**
  - Dependencies:
    - `EmbeddingRepository`.
    - `PlaceAnnIndexManager`.
    - `PlaceHead` (loaded with trained weights).
    - `PlaceRepository` (for metadata in explanations).
    - `ExplanationService`.
  - Method:
    - `recommend_places(request: PlaceRecommendationRequest) -> List[PlaceRecommendation]`
  - Steps:
    1. Fetch `z_u` from `EmbeddingRepository`. If missing, handle gracefully.
    2. Determine `city_id`:
       - Use `request.city_id` if provided.
       - Else, fall back to user’s home city.
    3. Use `PlaceAnnIndexManager` to get the city’s ANN index.
    4. Query the index with `z_u` to retrieve top `M` candidate places.
    5. For each candidate:
       - Fetch `z_p` from `EmbeddingRepository`.
       - Construct context vector from `request` (time slot, desired tags).
       - Compute `score = PlaceHead(z_u, z_p, ctx)`.
    6. Sort candidates by score, keep top `K = request.top_k`.
    7. For each top candidate:
       - Fetch `PlaceSchema` for metadata.
       - Call `ExplanationService.explain_place(...)`.
    8. Build and return `PlaceRecommendation` objects.

- **`PeopleRecommender`**
  - Dependencies:
    - `EmbeddingRepository`.
    - `UserAnnIndexManager`.
    - `FriendHead`.
    - `UserRepository`.
    - `PlaceRepository` (if target place is passed).
    - `ExplanationService`.
  - Method:
    - `recommend_people(request: PeopleRecommendationRequest) -> List[PeopleRecommendation]`
  - Steps:
    1. Fetch `z_u` for the query user.
    2. Determine `city_id` similarly to place recommender.
    3. Retrieve ANN user index for that city.
    4. Query with `z_u` to get top `M` candidate users.
    5. Filter out:
       - `user_id` itself.
       - Any blocked or disallowed users (if modeled).
    6. Build context:
       - If `target_place_id` given:
         - Optionally fetch place tags to include in `ctx_friend`.
       - Else use `activity_tags` or default context.
    7. For each candidate `v`:
       - Fetch `z_v`.
       - Compute `compat_score` and `attend_prob = FriendHead(z_u, z_v, ctx_friend)`.
       - Combine into `s_final`.
    8. Sort by `s_final`, keep top `K`.
    9. For each top candidate:
       - Fetch `UserSchema` for metadata.
       - Generate explanations via `ExplanationService.explain_people(...)`.
    10. Build and return `PeopleRecommendation` objects.

### 10.3 FastAPI Application (`serving/api_main.py`)

- Initialize app:
  - Load configuration.
  - Load `EmbeddingRepository` (embeddings into memory).
  - Load ANN indices from `INDEX_DIR`.
  - Load trained `PlaceHead` and `FriendHead` weights.
  - Initialize `PlaceRecommender`, `PeopleRecommender`, and `ExplanationService`.

- Define endpoints:
  - `GET /recommend/places`
    - Request: `PlaceRecommendationRequest`.
    - Response: `PlaceRecommendationResponse`.
    - Handler:
      - Calls `place_recommender.recommend_places(request)`.
  - `GET /recommend/people`
    - Request: `PeopleRecommendationRequest`.
    - Response: `PeopleRecommendationResponse`.
    - Handler:
      - Calls `people_recommender.recommend_people(request)`.

---

## 11. Explanation Service (`serving/explanations.py`)

### 11.1 `ExplanationService`

Responsibilities:

- Generate short, human-friendly explanation strings for:
  - **Place recommendations**.
  - **People recommendations**.

Dependencies:

- `UserRepository` and `PlaceRepository`.
- Access to:
  - User fine-tag preferences and vibe tags.
  - Place fine-tag vectors and categories.
  - Area/neighborhood information.

Key methods:

- **`explain_place(user: UserSchema, place: PlaceSchema, z_u: np.ndarray, z_p: np.ndarray, ctx) -> List[str]`**
  - Steps:
    1. Compute top-k overlapping fine tags between `user_fine_pref` and `place.fine_tag_vector`.
    2. Identify overlapping coarse categories where both have high weights.
    3. Determine if user frequently visits the same neighborhood as the place.
    4. Choose 1–3 explanation templates, such as:
       - “Matches your interest in `{tag1}` and `{tag2}`.”
       - “You often go out in `{neighborhood}`, and this place is nearby.”
       - “You like `{category}`, and this spot is highly rated for it.”

- **`explain_people(user_u: UserSchema, user_v: UserSchema, ctx, z_u: np.ndarray, z_v: np.ndarray) -> List[str]`**
  - Steps:
    1. Compute top overlapping **vibe tags**.
    2. Compute top overlapping fine-grained interests.
    3. Check shared neighborhoods (where both have high `area_freqs`).
    4. Choose templates, such as:
       - “You both like `{tag1}` and `{tag2}`.”
       - “You both often go out in `{neighborhood}`.”
       - “You both enjoy `{category}` places.”

The explanations are derived from features already in the model, ensuring they are **faithful** to the underlying signals.

---

## 12. End-to-End Execution Flow

### 12.1 Offline Flow

1. **Generate synthetic data**:
   - Run `scripts/run_synthetic_generation.py`:
     - Creates `users`, `places`, `interactions`, `user_user_edges`, `friend_labels`.
2. **Build features and graph**:
   - Run `scripts/run_build_features.py`:
     - Uses `user_features`, `place_features`, `interaction_features`, and `graph_builder`.
     - Produces `HeteroData` graph and ID mappings.
3. **Train GNN**:
   - Run `scripts/run_train_gnn.py`:
     - Trains `UserEncoder`, `PlaceEncoder`, `GraphRecBackbone`, `PlaceHead`, and `FriendHead`.
     - Saves model checkpoints and training logs.
4. **Export embeddings**:
   - Run `scripts/run_export_embeddings.py`:
     - Produces user and place embedding tables.
5. **Build ANN indices**:
   - Run `scripts/run_build_indices.py`:
     - Builds per-city ANN indices for users and places.

### 12.2 Online Flow

1. **Service startup**:
   - Run the FastAPI app (`api_main.py`):
     - Loads config.
     - Loads embeddings into memory.
     - Loads ANN indices.
     - Initializes recommender and explanation services.
2. **Place recommendation request**:
   - Client calls `/recommend/places` with `user_id` and optional context.
   - Service:
     - Looks up user embedding.
     - Gets city-specific place index and ANN candidates.
     - Scores via `PlaceHead`.
     - Generates explanations.
     - Returns top `K` recommendations.
3. **People recommendation request**:
   - Client calls `/recommend/people`.
   - Service:
     - Looks up user embedding.
     - Gets city-specific user index and candidates.
     - Scores via `FriendHead` (compatibility + attendance).
     - Generates explanations.
     - Returns top `K` candidate users.

---

## 13. Implementation Notes and Priorities

- **First milestone**:
  - Implement synthetic generation.
  - Implement feature and graph builder.
  - Train a first working GNN on synthetic data.

- **Second milestone**:
  - Implement embedding export and ANN indices.
  - Implement a basic FastAPI service with both endpoints wired to the trained model.

- **Third milestone**:
  - Improve explanations.
  - Tune hyperparameters and sampling strategies.
  - Add monitoring and logging for online recommendation quality.

This LLD is intentionally detailed so you can go module by module and implement with minimal ambiguity.


