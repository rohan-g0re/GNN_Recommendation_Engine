# Installation Notes - Windows ARM64

## Issue
Running on **Windows ARM64** (Python 3.11.9 ARM64). Many packages don't have pre-built wheels for ARM64 Windows and require compilation from source, which needs Visual Studio Build Tools.

## Current Status
- ✅ **numpy** - Installed (has ARM64 wheels)
- ✅ **requests** - Installed  
- ⚠️ **pandas** - Needs compilation or alternative
- ⚠️ **pyarrow** - Needs compilation
- ⚠️ **torch** - May need special installation
- ⚠️ **torch-geometric** - Depends on torch

## Solutions

### Option 1: Use Conda (Recommended for ARM64)
```powershell
# Install Miniconda for ARM64, then:
conda create -n nico_assgn python=3.11
conda activate nico_assgn
conda install pandas pyarrow numpy
pip install torch torch-geometric fastapi uvicorn pydantic faiss-cpu
```

### Option 2: Install Visual Studio Build Tools
1. Download Visual Studio Build Tools
2. Install "C++ build tools" workload
3. Then: `pip install -r requirements.txt`

### Option 3: Use x64 Python (if possible)
Switch to x64 Python instead of ARM64 for better package compatibility.

## Quick Test Commands
```powershell
# Test what's installed
python -c "import numpy; print('numpy OK')"
python -c "import pandas; print('pandas OK')"  # Will fail until installed
```

