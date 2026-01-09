import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Color palette consistent with existing figures
colors_matmul = sns.color_palette("flare_r", 6)
colors_normalization = sns.color_palette("summer", 3)
colors_gelu = sns.color_palette("pink", 1)
colors_allreduce = sns.color_palette("Blues_r", 2)
colors = colors_matmul + colors_normalization + colors_gelu + colors_allreduce

# -------- Prefill (Initial computation) --------
values_simgpu = (
    pd.read_csv("transformer_A100_sim.csv", header=None, names=categories, index_col=None)
    .iloc[0]
    .tolist()
)
# HBF filename follows test_transformer.py output
values_simgpu_hbf = (
    pd.read_csv(
        "transformer_A100_HBF_sim_HBF.csv", header=None, names=categories, index_col=None
    )
    .iloc[0]
    .tolist()
)

plt.figure(figsize=(3, 2.8))
# A100
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_simgpu)):
    plt.bar(1, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
sum_a100 = bottom

# A100-HBF
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_simgpu_hbf)):
    plt.bar(2, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
sum_hbf = bottom

print(f"Init total latency (s): A100={sum_a100:.4f}, A100-HBF={sum_hbf:.4f}, speedup={sum_a100/sum_hbf:.3f}x")

plt.ylabel("Latency (s)")
plt.xticks([1, 2], ["Simulated\nA100", "Simulated\nA100-HBF"])
plt.tight_layout()
plt.savefig("figure5i_hbf.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)

# -------- Autoregression --------
values_ar_a100 = (
    pd.read_csv("transformerAR_A100_sim.csv", header=None, names=categories, index_col=None)
    .iloc[0]
    .tolist()
)
values_ar_hbf = (
    pd.read_csv(
        "transformerAR_A100_HBF_sim.csv", header=None, names=categories, index_col=None
    )
    .iloc[0]
    .tolist()
)

plt.figure(figsize=(3, 2.8))
# A100 (ms)
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_ar_a100)):
    value_ms = value * 1e3
    plt.bar(1, value_ms, bottom=bottom, color=colors[i], width=0.5)
    bottom += value_ms
sum_ar_a100 = bottom

# A100-HBF (ms)
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_ar_hbf)):
    value_ms = value * 1e3
    plt.bar(2, value_ms, bottom=bottom, color=colors[i], width=0.5)
    bottom += value_ms
sum_ar_hbf = bottom

print(
    f"AR total latency (ms): A100={sum_ar_a100:.2f}, A100-HBF={sum_ar_hbf:.2f}, speedup={sum_ar_a100/sum_ar_hbf:.3f}x"
)

plt.ylabel("Latency (ms)")
plt.xticks([1, 2], ["Simulated\nA100", "Simulated\nA100-HBF"])
plt.tight_layout()
plt.savefig("figure5k_hbf.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
