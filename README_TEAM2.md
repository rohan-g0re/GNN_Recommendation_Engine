# Team 2: API Serving Team - README

## 🎯 Mission

Build the FastAPI service that serves place and people recommendations using Team 1's trained GNN embeddings.

---

## ✅ What's Done

**ALL core serving components are implemented!** You can start testing immediately.

### Implemented Components

1. **API Endpoints** - `/recommend/places`, `/recommend/people`, `/health`
2. **ANN Indexing** - Faiss-based per-city indices for fast retrieval
3. **Core Recommenders** - Place and people recommendation logic
4. **Explanation Service** - Human-readable explanations
5. **Supporting Scripts** - Index builder, validation, testing

See `TEAM2_STATUS.md` for complete details.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install fastapi uvicorn pydantic pandas pyarrow faiss-cpu torch numpy
```

### 2. Test Structure (No Team 1 Data Needed)

```bash
python scripts/test_api_structure.py
```

This verifies:
- ✅ All modules can be imported
- ✅ Constants are correct
- ✅ API schemas work
- ✅ FastAPI app structure is correct

### 3. Test with Mock Data (Optional)

```bash
python scripts/test_with_mocks.py
```

Creates mock embeddings and tests all components.

---

## 📦 Integration with Team 1 (Day 10+)

### Step 1: Receive Team 1's Deliverables

Team 1 should provide:
- `data/embeddings/user_embeddings.parquet`
- `data/embeddings/place_embeddings.parquet`
- `models/final_model.pt`
- `data/user_id_mappings.pkl` (optional)
- `data/place_id_mappings.pkl` (optional)

### Step 2: Validate

```bash
python scripts/validate_team1_deliverables.py
```

### Step 3: Build Indices

```bash
python scripts/run_build_indices.py \
    --embeddings_dir data/embeddings \
    --output_dir data/indices \
    --data_dir data
```

### Step 4: Start Server

```bash
uvicorn recsys.serving.api_main:app --host 0.0.0.0 --port 8000
```

### Step 5: Test

```bash
# Health check
curl http://localhost:8000/health

# Place recommendations
curl -X POST http://localhost:8000/recommend/places \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "city_id": 2, "top_k": 10}'

# People recommendations  
curl -X POST http://localhost:8000/recommend/people \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "city_id": 2, "top_k": 10}'
```

---

## 📚 Key Files

### Core Implementation
- `recsys/serving/api_main.py` - FastAPI application
- `recsys/serving/recommender_core.py` - Core recommendation logic
- `recsys/serving/ann_index.py` - ANN indexing
- `recsys/serving/explanations.py` - Explanation service
- `recsys/serving/api_schemas.py` - API request/response models

### Scripts
- `scripts/run_build_indices.py` - Build ANN indices
- `scripts/validate_team1_deliverables.py` - Validate Team 1's files
- `scripts/test_api_structure.py` - Structure tests
- `scripts/test_with_mocks.py` - Mock data tests

### Documentation
- `TEAM2_STATUS.md` - Complete implementation status
- `TEAM2_QUICKSTART.md` - Quick start guide
- `INTEGRATION_CONTRACTS.md` - Integration specifications

---

## 🔍 API Reference

### POST `/recommend/places`

Get place recommendations for a user.

**Request:**
```json
{
  "user_id": 42,
  "city_id": 2,              // Optional
  "time_slot": 3,            // Optional, 0-5
  "desired_categories": [0, 2],  // Optional
  "top_k": 10
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "place_id": 1234,
      "score": 0.87,
      "explanations": ["...", "..."]
    }
  ]
}
```

### POST `/recommend/people`

Get people recommendations for a user.

**Request:**
```json
{
  "user_id": 42,
  "city_id": 2,              // Optional
  "target_place_id": 1234,   // Optional
  "activity_tags": [5, 10],  // Optional
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
      "explanations": ["...", "..."]
    }
  ]
}
```

---

## ⚙️ Configuration

All constants are in `recsys/config/constants.py` and must match Team 1 exactly:
- `D_MODEL = 128`
- `D_USER_RAW = 148`
- `D_PLACE_RAW = 114`
- `N_CITIES = 8`

See `INTEGRATION_CONTRACTS.md` for full specifications.

---

## 🐛 Troubleshooting

### "Module not found" errors
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### "File not found: data/embeddings/..."
- **Solution**: Wait for Team 1's deliverables or run `test_with_mocks.py` for testing

### "Index not found for city X"
- **Solution**: Run `scripts/run_build_indices.py` to build indices

### "Service not initialized" (503 error)
- **Solution**: Check that embeddings and model checkpoint exist, then restart server

---

## ✅ Status

**READY FOR INTEGRATION**

All components implemented. Can test structure independently. Waiting for Team 1's deliverables for full integration.

---

## 📞 Support

- **Main docs**: `gnn_plan.md` Section 6
- **Integration specs**: `INTEGRATION_CONTRACTS.md`
- **Status**: `TEAM2_STATUS.md`

