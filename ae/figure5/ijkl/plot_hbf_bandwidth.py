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

plt.figure(figsize=(15, 5))
# Prefill subplot
ax1 = plt.subplot(1, 2, 1)
if prefill_points:
    xs = [p[0] for p in prefill_points[2:]]
    ys = [p[1] for p in prefill_points[2:]]
    ax1.plot(xs, ys, marker="o")
    ax1.set_xlabel("Bandwidth (GB/s)")
    ax1.set_ylabel("Prefill Latency (s)")
    ax1.set_title("A100-HBF Prefill vs Bandwidth")
    # Y-axis range based on max bandwidth point
    max_bw = max(xs)
    # Find latency at max bandwidth
    y_at_max = next(v for (bw, v) in prefill_points if bw == max_bw)
    # Ticks: 1x..10x of y_at_max
    multiples = [k * y_at_max for k in range(1, 11)]
    ax1.set_ylim(0, multiples[-1])
    ax1.set_yticks(multiples)
    ax1.set_yticklabels([f"{v:.4g}" for v in multiples])
    # Add red ratio labels (1x..10x) next to ticks
    for k, v in enumerate(multiples, start=1):
        ax1.text(
            1.02,
            v,
            f"{k}x",
            color="red",
            va="center",
            transform=ax1.get_yaxis_transform(),
        )
    ax1.grid(axis="both")
else:
    ax1.text(0.5, 0.5, "No prefill data", ha="center")

# Decode subplot
ax2 = plt.subplot(1, 2, 2)
if decode_points:
    xs = [p[0] for p in decode_points[2:]]
    ys = [p[1] for p in decode_points[2:]]
    ax2.plot(xs, ys, marker="o")
    ax2.set_xlabel("Bandwidth (GB/s)")
    ax2.set_ylabel("Decode Latency (ms)")
    ax2.set_title("A100-HBF Decode vs Bandwidth")
    max_bw = max(xs)
    # Time at the maximum bandwidth
    y_at_max = next(v for (bw, v) in decode_points if bw == max_bw)
    multiples = [k * y_at_max for k in range(1, 11)]
    ax2.set_ylim(0, multiples[-1])
    ax2.set_yticks(multiples)
    ax2.set_yticklabels([f"{v:.4g}" for v in multiples])
    for k, v in enumerate(multiples, start=1):
        ax2.text(
            1.02,
            v,
            f"{k}x",
            color="red",
            va="center",
            transform=ax2.get_yaxis_transform(),
        )
    ax2.grid(axis="both")
else:
    ax2.text(0.5, 0.5, "No decode data", ha="center")

plt.tight_layout()
plt.savefig(
    "hbf_bandwidth_prefill_decode.svg", bbox_inches="tight", pad_inches=0.01, dpi=300
)
print("Saved hbf_bandwidth_prefill_decode.svg")
