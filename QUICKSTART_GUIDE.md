# Quick Start Guide: GNN Recommendation Engine

## 🚀 Get Started in 5 Minutes

This guide helps you understand the project structure and start implementing immediately.

---

## 📚 Document Guide

### Start Here (5 min read):
1. **`FINAL_DELIVERY_SUMMARY.md`** - Overview of what's been delivered
2. This document - Quick navigation guide

### For Understanding (20 min read):
3. **`product_idea.md`** - Business context and vision
4. **`task_details.md`** - Requirements

### For Implementation:

**If you're on the Training Team**:
- **Main doc**: `gnn_plan.md` - Your complete implementation guide
- **Reference**: `INTEGRATION_CONTRACTS.md` - What you must deliver

**If you're on the Serving/API Team**:
- **Main docs**: 
  - `gnn_plan.md` Section 6 (Inference & Serving)
  - `lld_recommendation_engine.md` Sections 10-11
- **Reference**: `INTEGRATION_CONTRACTS.md` - What you'll receive

---

## 🗂️ Project Structure

```
nico_assgn/
├── product_idea.md              # Vision document
├── task_details.md              # Requirements
├── gnn_plan.md                  # ⭐ Complete GNN implementation (2,856 lines)
├── lld_recommendation_engine.md # ⭐ LLD for serving (1,100+ lines)
├── INTEGRATION_CONTRACTS.md     # ⭐ Critical: Team interfaces
├── FINAL_DELIVERY_SUMMARY.md    # What's been delivered
├── QUICKSTART_GUIDE.md          # This file
└── IMPLEMENTATION_STATUS.md     # Detailed status

To be created:
recsys/                          # Python package
├── config/                      # Constants, ModelConfig
├── data/                        # Schemas, repositories
├── features/                    # Feature encoding, graph building
├── ml/                          # Models, training
├── serving/                     # FastAPI, ANN, explanations
└── scripts/                     # Execution scripts
```

---

## 🎯 What's Your Role?

### 👨‍💻 Training Team (GNN/ML)

**Your job**: Train the GNN and produce embeddings

**Key document**: `gnn_plan.md`

**Implementation sections** (copy-paste ready code):
- Section 2: Data Models → `recsys/data/schemas.py`
- Section 3: Graph Building → `recsys/features/graph_builder.py`
- Section 4: Model Architecture → `recsys/ml/models/`
- Section 5: Training → `recsys/ml/training/`
- Section 6.1: Embedding Export → `scripts/run_export_embeddings.py`

**What you deliver** (per INTEGRATION_CONTRACTS.md):
1. `models/final_model.pt` - Trained checkpoint
2. `data/embeddings/user_embeddings.parquet` - User embeddings
3. `data/embeddings/place_embeddings.parquet` - Place embeddings
4. `data/user_id_mappings.pkl` - ID to index mappings
5. `data/place_id_mappings.pkl` - ID to index mappings

**Critical constants to use**:
```python
from recsys.config.constants import (
    D_USER_RAW,    # 148
    D_PLACE_RAW,   # 114
    D_MODEL,       # 128
    C_COARSE,      # 6
    C_FINE,        # 100
    C_VIBE         # 30
)
```

**Start coding**:
1. Create `recsys/config/constants.py` - Copy from `gnn_plan.md` Section 2.1
2. Create `recsys/data/schemas.py` - Copy from `gnn_plan.md` Section 2.2-2.6
3. Create `recsys/features/graph_builder.py` - Copy from `gnn_plan.md` Section 3.2-3.3
4. Continue through sections 4-5

---

### 🌐 Serving/API Team

**Your job**: Build FastAPI service that serves recommendations

**Key documents**: 
- `gnn_plan.md` Section 6
- `lld_recommendation_engine.md` Sections 10-11

**Implementation sections** (copy-paste ready code):
- Section 6.2: ANN Indexing → `recsys/serving/ann_index.py`
- Section 6.3: API Schemas → `recsys/serving/api_schemas.py`
- Section 6.4: Core Logic → `recsys/serving/recommender_core.py`
- Section 6.5: Explanations → `recsys/serving/explanations.py`
- Section 6.6: FastAPI App → `recsys/serving/api_main.py`

**What you receive** (from training team):
1. Embeddings (Parquet files)
2. Model checkpoint (PyTorch)
3. ID mappings (Pickle)

**API Endpoints you implement**:
```python
POST /recommend/places
POST /recommend/people
GET /health
```

**Start coding**:
1. Create `recsys/serving/api_schemas.py` - Copy from `gnn_plan.md` Section 6.3
2. Create `recsys/serving/ann_index.py` - Copy from `gnn_plan.md` Section 6.2
3. Create `recsys/serving/recommender_core.py` - Copy from `gnn_plan.md` Section 6.4
4. Create `recsys/serving/api_main.py` - Copy from `gnn_plan.md` Section 6.6

---

## ⚡ Quick Commands

### Setup (Both Teams):
```bash
# Create project structure
mkdir -p recsys/{config,data,features,ml,serving,scripts,synthetic}
cd recsys

# Create __init__.py files
find . -type d -exec touch {}/__init__.py \;

# Install dependencies (example)
pip install torch torch-geometric pandas pyarrow faiss-cpu fastapi uvicorn pydantic
```

### Training Team:
```bash
# 1. Generate synthetic data
python scripts/run_synthetic_generation.py --output_dir data/

# 2. Build graph
python scripts/run_build_features.py --data_dir data/ --output_dir data/

# 3. Train model
python scripts/run_train_gnn.py \
    --data_dir data/ \
    --output_dir models/ \
    --epochs 50

# 4. Export embeddings
python scripts/run_export_embeddings.py \
    --checkpoint models/final_model.pt \
    --data_dir data/ \
    --output_dir data/embeddings/

# 5. Build indices
python scripts/run_build_indices.py \
    --embeddings_dir data/embeddings/ \
    --output_dir data/indices/
```

### Serving Team:
```bash
# After receiving training outputs:

# 1. Verify files exist
ls data/embeddings/user_embeddings.parquet
ls data/embeddings/place_embeddings.parquet
ls models/final_model.pt

# 2. Start API server
python -m uvicorn recsys.serving.api_main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload

# 3. Test endpoints
curl -X POST http://localhost:8000/recommend/places \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 42,
    "city_id": 2,
    "top_k": 10
  }'
```

---

## 🔍 Finding Code Quickly

### Need to know feature dimensions?
→ `INTEGRATION_CONTRACTS.md` Section 1

### Need data normalization formulas?
→ `INTEGRATION_CONTRACTS.md` Section 2

### Need model architecture details?
→ `gnn_plan.md` Section 4

### Need API contract?
→ `INTEGRATION_CONTRACTS.md` Section 4

### Need file format specs?
→ `INTEGRATION_CONTRACTS.md` Section 3

### Need complete Python implementation?
→ `gnn_plan.md` (search for the component)

---

## ⚠️ Critical Things to Know

### DO:
✅ Use exact dimensions from `INTEGRATION_CONTRACTS.md`
✅ Use exact normalization formulas
✅ Set `model.eval()` in serving
✅ Use ID mappings (never raw IDs as indices)
✅ Validate feature vectors sum to 1.0

### DON'T:
❌ Change feature dimensions without updating both docs
❌ Use different normalization in training vs serving
❌ Forget to load checkpoint in serving
❌ Access embeddings by raw ID (use mappings!)
❌ Skip validation

---

## 🧪 Testing Integration

### Test 1: Feature Encoding
```python
from recsys.data.schemas import UserSchema
from recsys.features.graph_builder import encode_user_features

user = UserSchema(...)  # Create test user
features = encode_user_features(user, config)

assert features.shape == (148,), f"Expected 148, got {features.shape}"
assert 0 <= features[143] <= 1.0, "Behavior stat not normalized"
print("✅ Feature encoding works!")
```

### Test 2: Model Loading
```python
import torch
from recsys.ml.models.heads import PlaceHead
from recsys.config.model_config import ModelConfig

checkpoint = torch.load("models/final_model.pt")
config = ModelConfig()
place_head = PlaceHead(config)
place_head.load_state_dict(checkpoint['place_head'])
place_head.eval()

print("✅ Model loads successfully!")
```

### Test 3: End-to-End
```python
# Full test in INTEGRATION_CONTRACTS.md Section 6.2
```

---

## 📞 Getting Help

### Integration Issues?
1. Check `INTEGRATION_CONTRACTS.md` first
2. Verify dimensions match
3. Check normalization formulas
4. Run integration test

### Implementation Questions?
1. **Training**: See `gnn_plan.md` sections 2-5
2. **Serving**: See `gnn_plan.md` section 6
3. **Interfaces**: See `INTEGRATION_CONTRACTS.md`

### Architecture Questions?
1. High-level: `product_idea.md`
2. Technical: `gnn_plan.md` Section 4

---

## 🎓 Learning Path

### Day 1: Understanding
- Read `FINAL_DELIVERY_SUMMARY.md`
- Skim `product_idea.md`
- Read your team's main document

### Day 2-3: Setup
- Create project structure
- Copy-paste schema implementations
- Set up dependencies

### Week 1: Core Implementation
- **Training**: Implement models + training loop
- **Serving**: Implement API + recommenders

### Week 2: Integration
- Training team delivers outputs
- Serving team integrates
- Run end-to-end tests

### Week 3: Testing & Deployment
- Load testing
- Bug fixes
- Deploy to staging

---

## 🚦 Status Indicators

| Component | Status | Ready to Code |
|-----------|--------|---------------|
| Data Schemas | ✅ Complete | Yes |
| Feature Encoding | ✅ Complete | Yes |
| GNN Model | ✅ Complete | Yes |
| Training Loop | ✅ Complete | Yes |
| Embedding Export | ✅ Complete | Yes |
| ANN Indexing | ✅ Complete | Yes |
| FastAPI Service | ✅ Complete | Yes |
| Explanations | ✅ Complete | Yes |
| Synthetic Data | 🔄 Design Ready | Needs Implementation |
| Integration Tests | ⏳ Pending | After Integration |

---

## 💡 Pro Tips

1. **Start with constants**: Copy `recsys/config/constants.py` first - everything depends on it

2. **Copy-paste is OK**: The code in the documents is meant to be copied. Just verify dimensions.

3. **Test early**: Don't wait for full implementation. Test each component as you build it.

4. **Use type hints**: All code has type hints. Your IDE will help catch errors.

5. **Read integration contracts**: Before any team handoff, read the relevant section.

---

## 📋 Checklist Before Coding

Training Team:
- [ ] Read `gnn_plan.md` sections 2-6
- [ ] Read `INTEGRATION_CONTRACTS.md`
- [ ] Understand what you must deliver
- [ ] Set up Python environment
- [ ] Copy constants and schemas

Serving Team:
- [ ] Read `gnn_plan.md` section 6
- [ ] Read `INTEGRATION_CONTRACTS.md`
- [ ] Understand what you'll receive
- [ ] Set up FastAPI environment
- [ ] Copy API schemas and core logic

---

**Ready to code?** Start with your team's main document and follow the implementation sections in order. The code is complete and copy-paste ready!

**Questions?** Check `INTEGRATION_CONTRACTS.md` first, then your main document.

**Good luck! 🚀**

