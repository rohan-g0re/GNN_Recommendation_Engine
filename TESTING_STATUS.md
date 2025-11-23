# Testing Status & Blockers

## Current Blocker: Windows ARM64 Package Installation

**Issue**: Running on Windows ARM64. Many packages (pandas, pyarrow, torch) don't have pre-built wheels and require compilation from source, which needs Visual Studio Build Tools.

## What's Installed ✅
- numpy 2.3.5
- requests 2.32.5
- pip, setuptools, wheel

## What's Missing ❌
- pandas (required for data generation, repositories, embeddings)
- pyarrow (required for parquet file I/O)
- torch (required for GNN models)
- torch-geometric (required for graph neural networks)
- fastapi, uvicorn (required for API)
- faiss-cpu (required for ANN indexing)

## Code Dependencies on pandas
- `recsys/scripts/run_synthetic_generation.py` - saves data to parquet
- `recsys/data/repositories.py` - loads parquet files
- `recsys/scripts/run_export_embeddings.py` - saves embeddings to parquet
- `recsys/serving/recommender_core.py` - loads embeddings from parquet

## Solutions

### Option 1: Install Visual Studio Build Tools (Recommended)
1. Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Install "Desktop development with C++" workload
3. Restart terminal
4. Run: `pip install -r requirements.txt`

### Option 2: Use Conda (Easier for ARM64)
```powershell
# Install Miniconda for Windows ARM64
conda create -n nico_assgn python=3.11
conda activate nico_assgn
conda install pandas pyarrow numpy
pip install torch torch-geometric fastapi uvicorn pydantic faiss-cpu
```

### Option 3: Switch to x64 Python
If possible, use x64 Python instead of ARM64 for better package compatibility.

## Next Steps After Installation
1. Test synthetic data generation
2. Test graph building
3. Test training pipeline
4. Test embedding export
5. Test ANN index building
6. Test FastAPI server

