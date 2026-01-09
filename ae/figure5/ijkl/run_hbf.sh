#!/usr/bin/env bash
set -euo pipefail

# Clean previous outputs in this folder
rm -f ae/figure5/ijkl/*.csv
rm -f ae/figure5/ijkl/*.pdf

# Generate simulated CSVs (A100 and A100-HBF), init and autoregression
python -m ae.figure5.ijkl.test_transformer --simgpu
python -m ae.figure5.ijkl.test_transformer --simgpu-hbf
python -m ae.figure5.ijkl.test_transformer --simgpu --init
python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --init

# Plot the two-way comparison
cd ae/figure5/ijkl
python plot_transformer_hbf.py
