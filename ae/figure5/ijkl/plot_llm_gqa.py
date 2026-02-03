import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

colors_matmul = sns.color_palette("flare_r", 6)
colors_normalization = sns.color_palette("summer", 3)
colors_gelu = sns.color_palette("pink", 1)
colors_allreduce = sns.color_palette("Blues_r", 2)
colors = colors_matmul + colors_normalization + colors_gelu + colors_allreduce


def read_breakdown(path: str):
    return (
        pd.read_csv(path, header=None, names=categories, index_col=None)
        .iloc[0]
        .tolist()
    )


def plot_ar_gqa_vs_mha():
    mha_path = "transformerAR_A100_sim_mha.csv"
    gqa_path = "transformerAR_A100_sim_gqa_kv8.csv"

    values_mha = read_breakdown(mha_path)
    values_gqa = read_breakdown(gqa_path)

    plt.figure(figsize=(3, 2.8))

    bottom = 0.0
    for i, value in enumerate(values_mha):
        plt.bar(1, value * 1e3, bottom=bottom, color=colors[i], width=0.5)
        bottom += value * 1e3
    total_mha = bottom

    bottom = 0.0
    for i, value in enumerate(values_gqa):
        plt.bar(2, value * 1e3, bottom=bottom, color=colors[i], width=0.5)
        bottom += value * 1e3
    total_gqa = bottom

    print("AR GQA / MHA latency:", total_gqa / total_mha)

    plt.ylabel("Latency (ms)")
    plt.xticks([1, 2], ["MHA", "GQA\n(KV=8)"])
    plt.tight_layout()
    plt.savefig("figure5_gqa_ar.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


if __name__ == "__main__":
    plot_ar_gqa_vs_mha()
