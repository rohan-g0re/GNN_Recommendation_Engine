# Team 2 (API Serving Team) - Quick Start Guide

## ✅ What's Already Done

All core serving components have been implemented! You can start testing immediately.

### Implemented Files

- ✅ `recsys/serving/api_schemas.py` - API request/response models
- ✅ `recsys/serving/ann_index.py` - Faiss-based ANN indexing
- ✅ `recsys/serving/recommender_core.py` - Core recommendation logic
- ✅ `recsys/serving/explanations.py` - Explanation service
- ✅ `recsys/serving/api_main.py` - FastAPI application
- ✅ `scripts/run_build_indices.py` - Index builder script
- ✅ `scripts/validate_team1_deliverables.py` - Validation script
- ✅ `scripts/test_with_mocks.py` - Mock data testing

---

## 🚀 Quick Start (Independent Testing)

### Step 1: Test with Mock Data

```bash
# Run the mock data test suite
python scripts/test_with_mocks.py
```

This will:
- Create mock embeddings
- Test EmbeddingStore
- Test ANN indexing
- Test API schemas
- Verify all components work independently

### Step 2: Test FastAPI App Structure

```bash
# Start the API server (will fail on startup without Team 1's models, but structure is ready)
uvicorn recsys.serving.api_main:app --reload --port 8000
```

The server will attempt to load models on startup. Without Team 1's deliverables, it will fail, but you can verify:
- ✅ API structure is correct
- ✅ Endpoints are defined
- ✅ Schemas are valid

---

## 📋 Integration Checklist (Day 10+)

### After Receiving Team 1's Deliverables:

1. **Validate Deliverables**
   ```bash
   python scripts/validate_team1_deliverables.py
   ```

2. **Build ANN Indices**
   ```bash
   python scripts/run_build_indices.py \
       --embeddings_dir data/embeddings \
       --output_dir data/indices \
       --data_dir data
   ```

3. **Start Server**
   ```bash
   uvicorn recsys.serving.api_main:app --host 0.0.0.0 --port 8000
   ```

4. **Test Endpoints**
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

## 📚 Key Files Reference

### API Endpoints

- **POST** `/recommend/places` - Get place recommendations
- **POST** `/recommend/people` - Get people recommendations  
- **GET** `/health` - Health check

### Configuration

- Constants: `recsys/config/constants.py`
- Model config: `recsys/config/model_config.py`
- Integration contracts: `INTEGRATION_CONTRACTS.md`

### Core Components

- **EmbeddingStore**: Loads embeddings from parquet files
- **PlaceRecommender**: Generates place recommendations using ANN + model head
- **PeopleRecommender**: Generates people recommendations using ANN + friend head
- **ExplanationService**: Generates human-readable explanations

---

## 🔧 Dependencies

### Required (for serving)

```bash
pip install fastapi uvicorn pydantic pandas pyarrow faiss-cpu torch numpy
```

### Optional (for testing)

```bash
pip install requests  # For API testing
```

---

## 📝 Notes

- The code includes stub repositories that will work until Team 1 provides real implementations
- Model heads (`PlaceHead`, `FriendHead`, `ContextEncoder`) must be provided by Team 1
- Embeddings must be in parquet format with exact schema (see `INTEGRATION_CONTRACTS.md`)
- All constants match `INTEGRATION_CONTRACTS.md` exactly

---

## 🐛 Troubleshooting

### "Module not found: recsys.ml.models.heads"
- **Solution**: This is expected until Team 1 delivers the model files. The API structure is ready.

### "File not found: data/embeddings/user_embeddings.parquet"
- **Solution**: Run `python scripts/test_with_mocks.py` to create test data, or wait for Team 1's deliverables.

### "Index not found for city X"
- **Solution**: Run `scripts/run_build_indices.py` to build indices from embeddings.

---

## 📖 Documentation

- **Main implementation**: `gnn_plan.md` Section 6
- **Integration contracts**: `INTEGRATION_CONTRACTS.md`
- **Status**: `IMPLEMENTATION_STATUS.md`

---

**Status**: ✅ Ready for independent testing. Waiting for Team 1's deliverables for full integration.

