#!/bin/bash

# Activate your Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fpanet_3.8

# Force use of correct libcuda (from driver, not from old toolkit)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib

# (Optional) Print diagnostics
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Torch CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
