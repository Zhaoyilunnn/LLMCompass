#!/usr/bin/env bash
set -euo pipefail

# Sweep bandwidth for A100_HBF only, prefill and decode
# Range: 50 GB/s to 2000 GB/s, step 50 GB/s

# Clean previous outputs in this folder (only bandwidth-tagged files)
rm -f ae/figure5/ijkl/transformer_A100_HBF_sim_HBF_bw*.csv || true
rm -f ae/figure5/ijkl/transformerAR_A100_HBF_sim_bw*.csv || true
rm -f ae/figure5/ijkl/hbf_bandwidth_prefill_decode.pdf || true

# Generate simulated CSVs for each bandwidth point
for ((gb = 39; gb <= 2039; gb += 50)); do
  bw=$(
    python - <<PY
print(${gb} * 1e9)
PY
  )
  echo "Running A100_HBF prefill and decode at bandwidth ${gb} GB/s"
  # Prefill (init)
  python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --init --bandwidth "${bw}" || exit 1
  # Decode (autoregression)
  python -m ae.figure5.ijkl.test_transformer --simgpu-hbf --bandwidth "${bw}" || exit 1
done

# Plot
cd ae/figure5/ijkl
python plot_hbf_bandwidth.py
