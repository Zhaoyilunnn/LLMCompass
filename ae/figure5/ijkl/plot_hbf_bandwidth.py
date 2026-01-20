import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Categories must match CSV column order
categories = [
    "Q_K_V",
    "Q_mul_K",
    "A_mul_V",
    "Wo_proj",
    "W1_proj",
    "W2_proj",
    "Softmax",
    "LayerNorm_MHA",
    "LayerNorm_FFN",
    "GeLU",
    "AllReduce_MHA",
    "AllReduce_FFN",
]

# Collect files with bandwidth tags
prefill_points = []  # (gbps, total_latency_s)
decode_points = []  # (gbps, total_latency_ms)

for fname in os.listdir("."):
    if fname.startswith("transformer_A100_HBF_sim_HBF_bw") and fname.endswith(".csv"):
        # extract GB/s from suffix: _bw{GBs}GBs
        try:
            tag = fname.split("_bw")[1].split("GBs")[0]
            gbps = float(tag)
            values = pd.read_csv(fname, header=None, names=categories).iloc[0].tolist()
            total_s = sum(values)
            prefill_points.append((gbps, total_s))
        except Exception:
            pass
    elif fname.startswith("transformerAR_A100_HBF_sim_bw") and fname.endswith(".csv"):
        try:
            tag = fname.split("_bw")[1].split("GBs")[0]
            gbps = float(tag)
            values = pd.read_csv(fname, header=None, names=categories).iloc[0].tolist()
            total_ms = sum(values) * 1e3
            decode_points.append((gbps, total_ms))
        except Exception:
            pass

prefill_points.sort(key=lambda x: x[0])
decode_points.sort(key=lambda x: x[0])

plt.figure(figsize=(5.5, 2.6))
plt.subplot(1, 2, 1)
if prefill_points:
    xs = [p[0] for p in prefill_points]
    ys = [p[1] for p in prefill_points]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Bandwidth (GB/s)")
    plt.ylabel("Prefill Latency (s)")
    plt.title("A100-HBF Prefill vs Bandwidth")
else:
    plt.text(0.5, 0.5, "No prefill data", ha="center")

plt.subplot(1, 2, 2)
if decode_points:
    xs = [p[0] for p in decode_points]
    ys = [p[1] for p in decode_points]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Bandwidth (GB/s)")
    plt.ylabel("Decode Latency (ms)")
    plt.title("A100-HBF Decode vs Bandwidth")
else:
    plt.text(0.5, 0.5, "No decode data", ha="center")

plt.tight_layout()
plt.savefig(
    "hbf_bandwidth_prefill_decode.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300
)
print("Saved hbf_bandwidth_prefill_decode.pdf")
