#!/usr/bin/env bash
set -euo pipefail

# Clean previous outputs in this folder
rm -f ae/figure5/ijkl/*.csv
rm -f ae/figure5/ijkl/*.pdf

# Generate simulated CSVs
# Baseline A100 and A100-HBF (with fixed IO write coeff = 20 for HBF)
python -m ae.figure5.ijkl.test_transformer --simgpu
python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --fixed-io-write-coeff 20
# Exclude fixed latency variants for A100
python -m ae.figure5.ijkl.test_transformer --simgpu --exclude-fixed-latency
# Prefill (init) runs
python -m ae.figure5.ijkl.test_transformer --simgpu --init
python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --init --fixed-io-write-coeff 20
python -m ae.figure5.ijkl.test_transformer --simgpu --exclude-fixed-latency --init

# Plot the three-way comparison (A100-excl vs A100 vs A100-HBF, coeff=20)
cd ae/figure5/ijkl
python plot_transformer_hbf_compare_write_coeff.py
