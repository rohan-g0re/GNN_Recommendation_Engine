# Integration Contracts Between GNN Training & Serving Teams

## Purpose

This document defines the **exact interface contracts** between the GNN training team and the serving/API team to ensure seamless integration.

---

## 1. Feature Dimensions (MUST MATCH EXACTLY)

### 1.1 Raw Feature Dimensions

```python
# CRITICAL: These must be identical in both training and serving code

# User features
D_USER_RAW = 148  # Breakdown:
#   - 2: city_id, neighborhood_id (will be embedded)
#   - 6: cat_pref (C_COARSE)
#   - 100: fine_pref (C_FINE)
#   - 30: vibe_pref (C_VIBE)
#   - 5: area_freqs (MAX_NEIGHBORHOODS_PER_USER)
#   - 5: behavior stats

# Place features  
D_PLACE_RAW = 114  # Breakdown:
#   - 2: city_id, neighborhood_id
#   - 6: category_one_hot (C_COARSE)
#   - 100: fine_tag_vector (C_FINE)
#   - 2: price_band, typical_time_slot
#   - 4: popularity metrics

# Edge features
D_EDGE_UP = 12  # User-place edge
D_EDGE_UU = 3   # User-user edge

# Model dimension
D_MODEL = 128  # Output of encoders and GNN
```

### 1.2 Taxonomy Dimensions

```python
C_COARSE = 6    # Coarse categories
C_FINE = 100    # Fine-grained tags
C_VIBE = 30     # Vibe/personality tags

N_CITIES = 8
N_NEIGHBORHOODS_PER_CITY = 15
N_PRICE_BANDS = 5
N_TIME_SLOTS = 6
```

---

## 2. Data Normalization Schemes (MUST MATCH EXACTLY)

### 2.1 Preference Vectors

**Rule**: All preference vectors MUST sum to 1.0

```python
# Training code (synthetic generation)
user.cat_pref = normalize_to_sum_one(raw_cat_weights)
user.fine_pref = normalize_to_sum_one(raw_fine_weights)
user.vibe_pref = normalize_to_sum_one(raw_vibe_weights)

# Validation
assert abs(sum(user.cat_pref) - 1.0) < 1e-5
assert abs(sum(user.fine_pref) - 1.0) < 1e-5
assert abs(sum(user.vibe_pref) - 1.0) < 1e-5
```

### 2.2 Behavioral Statistics

**Rule**: Normalize to [0, 1] range using these exact formulas

```python
# Feature encoding (must match in training & serving)
features.extend([
    min(user.avg_sessions_per_week / 10.0, 1.0),
    min(user.avg_views_per_session / 100.0, 1.0),
    min(user.avg_likes_per_session / 10.0, 1.0),
    min(user.avg_saves_per_session / 10.0, 1.0),
    min(user.avg_attends_per_month / 20.0, 1.0)
])
```

### 2.3 Popularity Metrics

**Rule**: Log-normalize using log1p and divide by 5.0

```python
# Place features
features.extend([
    np.log1p(place.base_popularity) / 5.0,
    np.log1p(place.avg_daily_visits) / 5.0,
    place.conversion_rate,  # Already [0, 1]
    place.novelty_score     # Already [0, 1]
])
```

### 2.4 Implicit Rating Computation

**Rule**: Use this EXACT formula (defined in Section 2.1.1 of GNN plan)

```python
def compute_implicit_rating(
    dwell_time: float,
    num_likes: int,
    num_saves: int,
    num_shares: int,
    attended: bool
) -> float:
    score = 1.0
    score += min(dwell_time / 150.0, 2.0)
    score += min(num_likes * 0.5, 1.5)
    score += min(num_saves * 1.0, 2.0)
    score += min(num_shares * 0.5, 1.0)
    if attended:
        score += 2.0
    return min(score, 5.0)
```

---

## 3. File Formats & Persistence

### 3.1 Embedding Files

**Format**: Parquet with exact schema

```python
# user_embeddings.parquet
columns = [
    'user_id',      # int64
    'embedding'     # list<float> of length D_MODEL=128
]

# place_embeddings.parquet
columns = [
    'place_id',     # int64
    'embedding'     # list<float> of length D_MODEL=128
]
```

**Location**: `data/embeddings/`

**Loading in serving**:
```python
import pandas as pd

user_df = pd.read_parquet("data/embeddings/user_embeddings.parquet")
for _, row in user_df.iterrows():
    embedding_store.user_embeddings[row['user_id']] = np.array(row['embedding'])
```

### 3.2 Model Checkpoint

**Format**: PyTorch state dict

```python
checkpoint = {
    'user_encoder': user_encoder.state_dict(),
    'place_encoder': place_encoder.state_dict(),
    'backbone': backbone.state_dict(),
    'place_head': place_head.state_dict(),
    'friend_head': friend_head.state_dict(),
    'place_ctx_encoder': place_ctx_encoder.state_dict(),
    'friend_ctx_encoder': friend_ctx_encoder.state_dict(),
    'config': ModelConfig(...),  # As dict or object
    'optimizer': optimizer.state_dict()  # Optional
}

torch.save(checkpoint, "models/final_model.pt")
```

**Required for serving**: `place_head`, `friend_head`, `place_ctx_encoder`, `friend_ctx_encoder`

### 3.3 ID Mappings

**Format**: Pickle files

```python
# user_id_mappings.pkl
{
    'id_to_index': {user_id: graph_index, ...},
    'index_to_id': {graph_index: user_id, ...}
}

# place_id_mappings.pkl
{
    'id_to_index': {place_id: graph_index, ...},
    'index_to_id': {graph_index: place_id, ...}
}
```

**Location**: `data/`

**Critical**: IDs must map to the same graph indices used during training.

### 3.4 ANN Index Files

**Format**: Faiss index (pickled)

```python
# Structure per city
{
    'dimension': 128,
    'metric': 'cosine',
    'ids': [id1, id2, ...],  # Actual user/place IDs
    'index': faiss.serialize_index(index)
}
```

**Naming**: `indices/place_city_{city_id}.idx`, `indices/user_city_{city_id}.idx`

---

## 4. API Contracts

### 4.1 Place Recommendation Request

```python
{
    "user_id": 42,
    "city_id": 2,  # Optional, defaults to user's home city
    "time_slot": 3,  # 0-5, optional
    "desired_categories": [0, 2],  # List of category indices, optional
    "top_k": 10
}
```

### 4.2 Place Recommendation Response

```python
{
    "recommendations": [
        {
            "place_id": 1234,
            "score": 0.87,
            "explanations": [
                "Matches your interest in fishing and live music.",
                "You often go out in this neighborhood."
            ]
        },
        ...
    ]
}
```

### 4.3 People Recommendation Request

```python
{
    "user_id": 42,
    "city_id": 2,  # Optional
    "target_place_id": 1234,  # Optional, for context
    "activity_tags": [5, 10],  # Optional fine tag indices
    "top_k": 10
}
```

### 4.4 People Recommendation Response

```python
{
    "recommendations": [
        {
            "user_id": 789,
            "compat_score": 0.82,
            "attend_prob": 0.75,
            "combined_score": 0.799,  # alpha * compat + (1-alpha) * attend
            "explanations": [
                "You both like fishing and board games.",
                "You both often go out in the same neighborhoods."
            ]
        },
        ...
    ]
}
```

---

## 5. Model Architecture Contracts

### 5.1 UserEncoder Input/Output

```python
# Input
x_user: torch.Tensor  # Shape: (N_users, 148)

# Output  
encoded: torch.Tensor  # Shape: (N_users, 128)
```

**Feature layout in x_user**:
- `[0:1]`: city_id
- `[1:2]`: neighborhood_id
- `[2:8]`: cat_pref (6 dims)
- `[8:108]`: fine_pref (100 dims)
- `[108:138]`: vibe_pref (30 dims)
- `[138:143]`: area_freqs (5 dims)
- `[143:148]`: behavior stats (5 dims)

### 5.2 PlaceEncoder Input/Output

```python
# Input
x_place: torch.Tensor  # Shape: (N_places, 114)

# Output
encoded: torch.Tensor  # Shape: (N_places, 128)
```

**Feature layout in x_place**:
- `[0:1]`: city_id
- `[1:2]`: neighborhood_id
- `[2:8]`: category_one_hot (6 dims)
- `[8:108]`: fine_tag_vector (100 dims)
- `[108:109]`: price_band
- `[109:110]`: typical_time_slot
- `[110:114]`: popularity metrics (4 dims)

### 5.3 PlaceHead Input/Output

```python
# Input
z_user: torch.Tensor     # Shape: (batch, 128)
z_place: torch.Tensor    # Shape: (batch, 128)
ctx: torch.Tensor        # Shape: (batch, 16)

# Output
scores: torch.Tensor     # Shape: (batch,)
```

### 5.4 FriendHead Input/Output

```python
# Input
z_user_u: torch.Tensor   # Shape: (batch, 128)
z_user_v: torch.Tensor   # Shape: (batch, 128)
ctx: torch.Tensor        # Shape: (batch, 16)

# Output
compat_logits: torch.Tensor  # Shape: (batch,)
attend_prob: torch.Tensor    # Shape: (batch,), range [0, 1]
```

---

## 6. Testing & Validation

### 6.1 Unit Test Checklist

**Training team MUST provide**:
- [ ] Sample user_embeddings.parquet (10 users)
- [ ] Sample place_embeddings.parquet (10 places)
- [ ] Sample checkpoint.pt with trained heads
- [ ] Sample ID mappings (user_id_mappings.pkl, place_id_mappings.pkl)

**Serving team MUST validate**:
- [ ] Can load all embeddings
- [ ] Can load checkpoint and instantiate heads
- [ ] Can build ANN indices from embeddings
- [ ] API returns valid responses for test user_ids

### 6.2 Integration Test

```python
def test_end_to_end_integration():
    """
    Test that serving can consume training outputs.
    """
    # 1. Load embeddings
    embedding_store = EmbeddingStore()
    embedding_store.load_from_parquet(
        "data/embeddings/user_embeddings.parquet",
        "data/embeddings/place_embeddings.parquet"
    )
    
    # 2. Load checkpoint
    checkpoint = torch.load("models/final_model.pt")
    place_head = PlaceHead(ModelConfig())
    place_head.load_state_dict(checkpoint['place_head'])
    
    # 3. Test inference
    user_id = 42
    z_u = embedding_store.get_user_embedding(user_id)
    assert z_u.shape == (128,)
    
    # 4. Test scoring
    place_id = 100
    z_p = embedding_store.get_place_embedding(place_id)
    
    z_u_torch = torch.tensor(z_u).unsqueeze(0)
    z_p_torch = torch.tensor(z_p).unsqueeze(0)
    ctx = torch.zeros(1, 16)
    
    with torch.no_grad():
        score = place_head(z_u_torch, z_p_torch, ctx)
    
    assert score.shape == (1,)
    print("✅ Integration test passed!")
```

---

## 7. Deployment Checklist

### Training Team Deliverables

- [ ] `data/embeddings/user_embeddings.parquet`
- [ ] `data/embeddings/place_embeddings.parquet`
- [ ] `models/final_model.pt` (checkpoint with all heads)
- [ ] `data/user_id_mappings.pkl`
- [ ] `data/place_id_mappings.pkl`
- [ ] `data/users.parquet` (metadata for serving)
- [ ] `data/places.parquet` (metadata for serving)
- [ ] Documentation: feature normalization schemes used

### Serving Team Deliverables

- [ ] ANN index builder script
- [ ] FastAPI application (`recsys/serving/api_main.py`)
- [ ] Explanation service implementation
- [ ] Health check endpoint
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Load testing results

---

## 8. Common Pitfalls & Solutions

### Pitfall 1: Dimension Mismatch

**Symptom**: `RuntimeError: size mismatch` when loading checkpoint

**Solution**: Verify ALL dimension constants match:
```bash
# In training code
grep -r "D_USER_RAW\|D_PLACE_RAW\|D_MODEL" recsys/

# In serving code  
grep -r "D_USER_RAW\|D_PLACE_RAW\|D_MODEL" recsys/
```

### Pitfall 2: Normalization Inconsistency

**Symptom**: Embeddings work in training but poor results in serving

**Solution**: Add validation in serving:
```python
assert abs(sum(user.cat_pref) - 1.0) < 1e-5
assert 0 <= features[143] <= 1.0  # Normalized behavior stat
```

### Pitfall 3: ID Mapping Confusion

**Symptom**: Can't find embeddings for user/place IDs

**Solution**: Always use ID mappings:
```python
# DON'T use raw IDs as array indices
z_u = embeddings[user_id]  # WRONG!

# DO use mappings
user_idx = user_id_to_index[user_id]
z_u = embeddings[user_idx]  # CORRECT
```

### Pitfall 4: Model Mode

**Symptom**: Inconsistent predictions, dropout causing randomness

**Solution**: Always set eval mode in serving:
```python
place_head.eval()
friend_head.eval()

with torch.no_grad():
    score = place_head(z_u, z_p, ctx)
```

---

## 9. Version Control

**Rule**: Any change to constants/dimensions requires updating BOTH documents:
1. `gnn_plan.md` (training specs)
2. `lld_recommendation_engine.md` (serving specs)

**Version tracking**:
```python
# recsys/config/version.py
MODEL_VERSION = "1.0.0"
FEATURE_VERSION = "1.0.0"  # Increment if dimensions/normalization changes
API_VERSION = "1.0.0"
```

---

## 10. Contact & Escalation

**Training Team Lead**: [Name]
**Serving Team Lead**: [Name]

**For integration issues**:
1. Check this document first
2. Verify dimensions and normalization
3. Run integration test
4. If still blocked, escalate to leads

---

This contract ensures both teams can work independently while guaranteeing seamless integration.

