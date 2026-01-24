#!/usr/bin/env bash
set -euo pipefail

# Sweep sequence length for A100 and A100_HBF, prefill and decode
# Points: 2048, 4096, 2048*4, 2048*8

# Clean previous sequence-length outputs in this folder
rm -f ae/figure5/ijkl/transformer_A100_sim_seq*.csv || true
rm -f ae/figure5/ijkl/transformerAR_A100_sim_seq*.csv || true
rm -f ae/figure5/ijkl/transformer_A100_HBF_sim_HBF_seq*.csv || true
rm -f ae/figure5/ijkl/transformerAR_A100_HBF_sim_seq*.csv || true
rm -f ae/figure5/ijkl/hbf_seq_len_prefill_decode.svg || true

# Generate simulated CSVs for each sequence length point
for s in 2048 4096 $((2048*4)) $((2048*8)); do
  echo "Running A100 and A100_HBF prefill and decode at seq-len ${s}"
  # Prefill (init) A100 baseline
  python -m ae.figure5.ijkl.test_transformer --simgpu --init --seq-len "${s}" || exit 1
  # Prefill (init) A100_HBF
  python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --init --seq-len "${s}" || exit 1
  # Decode (autoregression) A100 baseline
  python -m ae.figure5.ijkl.test_transformer --simgpu --seq-len "${s}" || exit 1
  # Decode (autoregression) A100_HBF
  python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --seq-len "${s}" || exit 1
done

# Plot
cd ae/figure5/ijkl
python plot_hbf_seq_len.py
