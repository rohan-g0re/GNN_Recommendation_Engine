# Team 2 (API Serving Team) - Implementation Status

## ✅ COMPLETE: All Core Components Implemented

### 1. API Schemas (`recsys/serving/api_schemas.py`)
- ✅ `PlaceRecommendationRequest` - Request model for place recommendations
- ✅ `PlaceRecommendationResponse` - Response model with recommendations list
- ✅ `PeopleRecommendationRequest` - Request model for people recommendations  
- ✅ `PeopleRecommendationResponse` - Response model with people recommendations
- ✅ All models use Pydantic for validation

### 2. ANN Indexing (`recsys/serving/ann_index.py`)
- ✅ `AnnIndex` class - Faiss wrapper for cosine similarity search
- ✅ `CityAnnIndexManager` - Per-city index management
- ✅ Save/load functionality for indices
- ✅ Supports both user and place embeddings

### 3. Core Recommenders (`recsys/serving/recommender_core.py`)
- ✅ `EmbeddingStore` - Loads embeddings from parquet files
- ✅ `PlaceRecommender` - Complete place recommendation logic:
  - ANN candidate retrieval
  - Context encoding (city, time_slot, categories)
  - Model head scoring
  - Explanation generation
- ✅ `PeopleRecommender` - Complete people recommendation logic:
  - ANN candidate retrieval
  - Compatibility + attendance probability scoring
  - Combined score calculation
  - Explanation generation

### 4. Explanation Service (`recsys/serving/explanations.py`)
- ✅ `ExplanationService` class
- ✅ `explain_place()` - Generates explanations based on:
  - Fine tag overlaps
  - Category alignment
  - Neighborhood proximity
- ✅ `explain_people()` - Generates explanations based on:
  - Vibe/personality overlaps
  - Fine interest overlaps
  - Shared neighborhoods

### 5. FastAPI Application (`recsys/serving/api_main.py`)
- ✅ FastAPI app with CORS middleware
- ✅ `POST /recommend/places` endpoint
- ✅ `POST /recommend/people` endpoint
- ✅ `GET /health` endpoint
- ✅ Startup event to load:
  - Embeddings from parquet
  - ANN indices
  - Model checkpoint (heads + context encoders)
  - Repositories
- ✅ Error handling and validation

### 6. Supporting Scripts
- ✅ `scripts/run_build_indices.py` - Builds ANN indices from embeddings
- ✅ `scripts/validate_team1_deliverables.py` - Validates Team 1's deliverables
- ✅ `scripts/test_with_mocks.py` - Test suite with mock data
- ✅ `scripts/test_api_structure.py` - Structure validation tests

### 7. Documentation
- ✅ `requirements.txt` - All dependencies listed
- ✅ `TEAM2_QUICKSTART.md` - Quick start guide
- ✅ `INTEGRATION_CONTRACTS.md` - Integration specifications (from Team 1)

---

## 🔧 Setup Instructions

### Step 1: Install Dependencies

```bash
# Activate virtual environment (if using)
# Windows PowerShell:
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Or install directly:
pip install fastapi uvicorn pydantic pandas pyarrow faiss-cpu torch numpy
```

### Step 2: Verify Structure

```bash
# Test that all modules can be imported (after installing dependencies)
python scripts/test_api_structure.py
```

### Step 3: Test with Mocks (Optional)

```bash
# Create mock embeddings and test components
python scripts/test_with_mocks.py
```

---

## 📋 Integration Checklist (Day 10+)

### After Receiving Team 1's Deliverables:

- [ ] **Validate Deliverables**
  ```bash
  python scripts/validate_team1_deliverables.py
  ```
  Expected files:
  - `data/embeddings/user_embeddings.parquet`
  - `data/embeddings/place_embeddings.parquet`
  - `models/final_model.pt`
  - `data/user_id_mappings.pkl` (optional)
  - `data/place_id_mappings.pkl` (optional)

- [ ] **Build ANN Indices**
  ```bash
  python scripts/run_build_indices.py \
      --embeddings_dir data/embeddings \
      --output_dir data/indices \
      --data_dir data
  ```
  This creates:
  - `data/indices/user_city_0.idx` through `user_city_7.idx`
  - `data/indices/place_city_0.idx` through `place_city_7.idx`

- [ ] **Start API Server**
  ```bash
  uvicorn recsys.serving.api_main:app --host 0.0.0.0 --port 8000
  ```

- [ ] **Test Endpoints**
  ```bash
  # Health check
  curl http://localhost:8000/health
  # Should return: {"status": "healthy"}
  
  # Place recommendations
  curl -X POST http://localhost:8000/recommend/places \
    -H "Content-Type: application/json" \
    -d '{"user_id": 42, "city_id": 2, "time_slot": 3, "top_k": 10}'
  
  # People recommendations
  curl -X POST http://localhost:8000/recommend/people \
    -H "Content-Type: application/json" \
    -d '{"user_id": 42, "city_id": 2, "top_k": 10}'
  ```

---

## 🎯 API Endpoints Reference

### POST `/recommend/places`

**Request:**
```json
{
  "user_id": 42,
  "city_id": 2,              // Optional, defaults to user's home city
  "time_slot": 3,            // Optional, 0-5
  "desired_categories": [0, 2],  // Optional, list of category indices
  "top_k": 10               // Number of recommendations
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "place_id": 1234,
      "score": 0.87,
      "explanations": [
        "Matches your interest in fishing and live music.",
        "You often go out in this neighborhood."
      ]
    }
  ]
}
```

### POST `/recommend/people`

**Request:**
```json
{
  "user_id": 42,
  "city_id": 2,              // Optional
  "target_place_id": 1234,  // Optional, for context
  "activity_tags": [5, 10],  // Optional, fine tag indices
  "top_k": 10
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "user_id": 789,
      "compat_score": 0.82,
      "attend_prob": 0.75,
      "combined_score": 0.799,
      "explanations": [
        "You both like fishing and board games.",
        "You both often go out in the same neighborhoods."
      ]
    }
  ]
}
```

### GET `/health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

## 🔍 Key Implementation Details

### Constants (Must Match Team 1)
- `D_MODEL = 128` - Embedding dimension
- `D_USER_RAW = 148` - User feature dimension
- `D_PLACE_RAW = 114` - Place feature dimension
- `N_CITIES = 8` - Number of cities
- `C_COARSE = 6` - Coarse categories
- `C_FINE = 100` - Fine-grained tags
- `C_VIBE = 30` - Vibe/personality tags

### Model Loading
- Loads `place_head`, `friend_head`, `place_ctx_encoder`, `friend_ctx_encoder` from checkpoint
- Sets all models to `eval()` mode for inference
- Uses `torch.no_grad()` for all predictions

### ANN Indexing
- Uses Faiss with cosine similarity (normalized L2 + inner product)
- Separate indices per city for both users and places
- Saves indices as pickle files with serialized Faiss index

### Error Handling
- Returns 503 if service not initialized
- Returns 500 for other errors with detail message
- Validates all inputs via Pydantic schemas

---

## 📝 Notes

- **Repositories**: Currently uses stub implementations. Will work with Team 1's real repositories when provided.
- **Model Heads**: Must be provided by Team 1 in `recsys/ml/models/heads.py`
- **Embeddings**: Must be in parquet format with exact schema (see `INTEGRATION_CONTRACTS.md`)
- **Indices**: Built per-city for efficient retrieval

---

## ✅ Status: READY FOR INTEGRATION

All serving components are implemented and ready. Waiting for Team 1's deliverables to complete integration.

**Next Action**: Install dependencies and test structure, then wait for Team 1's embeddings and model checkpoint.

