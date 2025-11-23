# GNN Training Pipeline - Quick Start Guide

This guide helps you run the complete GNN training pipeline and generate all required deliverables.

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Steps

### Step 1: Generate Synthetic Data

```bash
python recsys/scripts/run_synthetic_generation.py \
    --output_dir data/ \
    --n_users 10000 \
    --n_places 10000
```

This creates:
- `data/users.parquet`
- `data/places.parquet`
- `data/interactions.parquet`
- `data/user_user_edges.parquet`
- `data/friend_labels.parquet`

### Step 2: Build Graph

```bash
python recsys/scripts/run_build_features.py \
    --data_dir data/ \
    --output_dir data/
```

This creates:
- `data/hetero_graph.pt`
- `data/user_id_mappings.pkl`
- `data/place_id_mappings.pkl`

### Step 3: Train Model

```bash
python recsys/scripts/run_train_gnn.py \
    --data_dir data/ \
    --output_dir models/ \
    --epochs 50 \
    --device cuda
```

This creates:
- `models/final_model.pt` (checkpoint with all model components)

### Step 4: Export Embeddings

```bash
python recsys/scripts/run_export_embeddings.py \
    --checkpoint models/final_model.pt \
    --data_dir data/ \
    --output_dir data/embeddings/
```

This creates:
- `data/embeddings/user_embeddings.parquet`
- `data/embeddings/place_embeddings.parquet`

## Deliverables Checklist

After running all steps, you should have:

✅ `models/final_model.pt` - Trained model checkpoint  
✅ `data/embeddings/user_embeddings.parquet` - User embeddings  
✅ `data/embeddings/place_embeddings.parquet` - Place embeddings  
✅ `data/user_id_mappings.pkl` - User ID mappings  
✅ `data/place_id_mappings.pkl` - Place ID mappings  
✅ `data/users.parquet` - User metadata  
✅ `data/places.parquet` - Place metadata  

## Key Dimensions (MUST MATCH)

- `D_USER_RAW = 148`
- `D_PLACE_RAW = 114`
- `D_MODEL = 128`
- All preference vectors sum to 1.0

## Troubleshooting

- If you get CUDA errors, use `--device cpu` in training script
- Ensure all parquet files are generated before building graph
- Check that graph is built before training

