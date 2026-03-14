import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from tensorflow.keras.models import load_model
import os

def main():
    parser = argparse.ArgumentParser(description="Motif visualization")
    parser.add_argument("--model", required=True, help="Path to model (.keras)")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    model_path = args.model
    out_dir = args.out

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"📌 Loading model: {model_path}")
    model = load_model(model_path, compile=False)

    # 找第一層 Conv1D
    conv_layer = None
    for layer in model.layers:
        if "conv1d" in layer.name.lower():
            conv_layer = layer
            break

    if conv_layer is None:
        raise ValueError("❌ 沒有找到 Conv1D 層！")

    filters, biases = conv_layer.get_weights()

    print(f"🧬 Conv layer found: {conv_layer.name}")
    print(f"Filters shape: {filters.shape}  (kernel_size, channels, num_filters)")

    # 只取前 6 個 motifs
    num_motifs = min(6, filters.shape[-1])
    print(f"🔍 Drawing first {num_motifs} motifs")

    # 固定 2x3 排版
    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(12, 6))
    axs = axs.flatten()

    for i in range(num_motifs):
        motif_matrix = filters[:, :, i]
        df = pd.DataFrame(motif_matrix, columns=["A", "C", "G", "T"])
        logomaker.Logo(df, ax=axs[i])
        axs[i].set_title(f"Motif #{i+1}", fontsize=12)

    # 隱藏多餘的框
    for j in range(num_motifs, 6):
        axs[j].axis("off")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "motifs.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"✅ Motif visualization saved to: {out_file}")


if __name__ == "__main__":
    main()