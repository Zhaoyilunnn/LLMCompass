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
values_a100_excl = (
    pd.read_csv(
        "transformer_A100_sim_excl.csv", header=None, names=categories, index_col=None
    )
    .iloc[0]
    .tolist()
)
values_a100 = (
    pd.read_csv(
        "transformer_A100_sim.csv", header=None, names=categories, index_col=None
    )
    .iloc[0]
    .tolist()
)
values_hbf_coeff33 = (
    pd.read_csv(
        "transformer_A100_HBF_sim_HBF_coeff33_lat3e-06.csv",
        header=None,
        names=categories,
        index_col=None,
    )
    .iloc[0]
    .tolist()
)

plt.figure(figsize=(4.8, 2.8))
# A100-excl
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_a100_excl)):
    plt.bar(1, value, bottom=bottom, color=colors[i], width=0.5, label=category)
    bottom += value
sum_a100_excl = bottom

# A100
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_a100)):
    plt.bar(2, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
sum_a100 = bottom

# HBF coeff33
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_hbf_coeff33)):
    plt.bar(3, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
sum_hbf_coeff33 = bottom

print(
    f"Init total (s): A100-excl={sum_a100_excl:.4f}, A100={sum_a100:.4f}, HBF_coeff33={sum_hbf_coeff33:.4f}"
)
print(
    f"Init speedups: excl/HBF_coeff33={sum_a100_excl / sum_hbf_coeff33:.3f}x, A100/HBF_coeff33={sum_a100 / sum_hbf_coeff33:.3f}x"
)

plt.ylabel("Latency (s)")
plt.xticks([1, 2, 3], ["A100\n(excl)", "A100", "A100-HBF"])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1.05))
plt.tight_layout()
plt.savefig(
    "figure5i_hbf_compare_coeff33.svg", bbox_inches="tight", pad_inches=0.01, dpi=300
)

# -------- Autoregression --------
values_ar_a100_excl = (
    pd.read_csv(
        "transformerAR_A100_sim_excl.csv", header=None, names=categories, index_col=None
    )
    .iloc[0]
    .tolist()
)
values_ar_a100 = (
    pd.read_csv(
        "transformerAR_A100_sim.csv", header=None, names=categories, index_col=None
    )
    .iloc[0]
    .tolist()
)
values_ar_hbf_coeff33 = (
    pd.read_csv(
        "transformerAR_A100_HBF_sim_coeff33_lat3e-06.csv",
        header=None,
        names=categories,
        index_col=None,
    )
    .iloc[0]
    .tolist()
)

plt.figure(figsize=(4.8, 2.8))
# A100-excl (ms)
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_ar_a100_excl)):
    value_ms = value * 1e3
    plt.bar(1, value_ms, bottom=bottom, color=colors[i], width=0.5, label=category)
    bottom += value_ms
sum_ar_a100_excl = bottom

# A100 (ms)
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_ar_a100)):
    value_ms = value * 1e3
    plt.bar(2, value_ms, bottom=bottom, color=colors[i], width=0.5)
    bottom += value_ms
sum_ar_a100 = bottom

# HBF coeff33 (ms)
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_ar_hbf_coeff33)):
    value_ms = value * 1e3
    plt.bar(3, value_ms, bottom=bottom, color=colors[i], width=0.5)
    bottom += value_ms
sum_ar_hbf_coeff33 = bottom

print(
    f"AR total (ms): A100-excl={sum_ar_a100_excl:.2f}, A100={sum_ar_a100:.2f}, HBF_coeff33={sum_ar_hbf_coeff33:.2f}"
)
print(
    f"AR speedups: excl/HBF_coeff33={sum_ar_a100_excl / sum_ar_hbf_coeff33:.3f}x, A100/HBF_coeff33={sum_ar_a100 / sum_ar_hbf_coeff33:.3f}x"
)

plt.ylabel("Latency (ms)")
plt.xticks([1, 2, 3], ["A100\n(excl)", "A100", "A100-HBF"])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1.05))
plt.tight_layout()
plt.savefig(
    "figure5k_hbf_compare_coeff33.svg", bbox_inches="tight", pad_inches=0.01, dpi=300
)
