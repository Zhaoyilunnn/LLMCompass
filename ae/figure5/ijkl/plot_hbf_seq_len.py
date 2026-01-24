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

# Collect files with sequence-length tags
# For each seq-len we collect both A100 (GPU baseline) and A100_HBF
prefill_gpu = {}  # seq_len -> total_latency_s
prefill_hbf = {}  # seq_len -> total_latency_s
decode_gpu = {}  # seq_len -> total_latency_ms
decode_hbf = {}  # seq_len -> total_latency_ms

for fname in os.listdir("."):
    # Prefill A100 baseline
    if fname.startswith("transformer_A100_sim_seq") and fname.endswith(".csv"):
        try:
            tag = fname.split("_seq")[1].split(".csv")[0]
            seq_len = int(tag)
            values = pd.read_csv(fname, header=None, names=categories).iloc[0].tolist()
            prefill_gpu[seq_len] = sum(values)
        except Exception:
            pass
    # Prefill A100_HBF
    elif fname.startswith("transformer_A100_HBF_sim_HBF_seq") and fname.endswith(
        ".csv"
    ):
        try:
            tag = fname.split("_seq")[1].split(".csv")[0]
            seq_len = int(tag)
            values = pd.read_csv(fname, header=None, names=categories).iloc[0].tolist()
            prefill_hbf[seq_len] = sum(values)
        except Exception:
            pass
    # Decode A100 baseline
    elif fname.startswith("transformerAR_A100_sim_seq") and fname.endswith(".csv"):
        try:
            tag = fname.split("_seq")[1].split(".csv")[0]
            seq_len = int(tag)
            values = pd.read_csv(fname, header=None, names=categories).iloc[0].tolist()
            decode_gpu[seq_len] = sum(values) * 1e3
        except Exception:
            pass
    # Decode A100_HBF
    elif fname.startswith("transformerAR_A100_HBF_sim_seq") and fname.endswith(".csv"):
        try:
            tag = fname.split("_seq")[1].split(".csv")[0]
            seq_len = int(tag)
            values = pd.read_csv(fname, header=None, names=categories).iloc[0].tolist()
            decode_hbf[seq_len] = sum(values) * 1e3
        except Exception:
            pass

# Common sorted seq-len keys where both gpu and hbf exist
prefill_keys = sorted(set(prefill_gpu.keys()) & set(prefill_hbf.keys()))
decode_keys = sorted(set(decode_gpu.keys()) & set(decode_hbf.keys()))

plt.figure(figsize=(10, 4))
# Prefill subplot
ax1 = plt.subplot(1, 2, 1)
if prefill_keys:
    xs = prefill_keys
    ys_gpu = [prefill_gpu[k] for k in xs]
    ys_hbf = [prefill_hbf[k] for k in xs]
    ax1.plot(xs, ys_gpu, marker="o", label="A100")
    ax1.plot(xs, ys_hbf, marker="s", label="A100-HBF")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Prefill Latency (s)")
    ax1.set_title("Prefill vs Sequence Length")
    ax1.grid(axis="both")
    # 标注 HBF 相比 GPU 的 overhead 比例
    for x, y_g, y_h in zip(xs, ys_gpu, ys_hbf):
        if y_g > 0:
            overhead = (y_h - y_g) / y_g
            ax1.text(
                x,
                y_h,
                f"{overhead * 100:.1f}%",
                fontsize=8,
                ha="center",
                va="bottom",
            )
    ax1.legend()
else:
    ax1.text(0.5, 0.5, "No prefill data", ha="center")

# Decode subplot
ax2 = plt.subplot(1, 2, 2)
if decode_keys:
    xs = decode_keys
    ys_gpu = [decode_gpu[k] for k in xs]
    ys_hbf = [decode_hbf[k] for k in xs]
    ax2.plot(xs, ys_gpu, marker="o", label="A100")
    ax2.plot(xs, ys_hbf, marker="s", label="A100-HBF")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Decode Latency (ms)")
    ax2.set_title("Decode vs Sequence Length")
    ax2.grid(axis="both")
    for x, y_g, y_h in zip(xs, ys_gpu, ys_hbf):
        if y_g > 0:
            overhead = (y_h - y_g) / y_g
            ax2.text(
                x,
                y_h,
                f"{overhead * 100:.1f}%",
                fontsize=8,
                ha="center",
                va="bottom",
            )
    ax2.legend()
else:
    ax2.text(0.5, 0.5, "No decode data", ha="center")

plt.tight_layout()
plt.savefig(
    "hbf_seq_len_prefill_decode.svg", bbox_inches="tight", pad_inches=0.01, dpi=300
)
print("Saved hbf_seq_len_prefill_decode.svg")
