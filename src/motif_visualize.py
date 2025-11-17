import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from tensorflow.keras.models import load_model

# === 輸入你的模型檔案路徑 ===
model_path = "./example/model_random.hdf5"

# === 載入模型 ===
model = load_model(model_path, compile=False)

print(f"✅ Model loaded from: {model_path}")

# === 找出第一層 Conv1D ===
conv_layer = None
for layer in model.layers:
    if "conv1d" in layer.name.lower():
        conv_layer = layer
        break

if conv_layer is None:
    raise ValueError("❌ 沒有找到 Conv1D 層，請檢查模型結構！")

filters, biases = conv_layer.get_weights()
print(f"🧬 Found Conv1D layer: {conv_layer.name}")
print(f"Filter shape: {filters.shape}  (length, channels, num_filters)")

# === 設定要畫的前 N 個 motifs ===
num_motifs = min(6, filters.shape[-1])  # 最多畫 6 個
motif_length = filters.shape[0]

# === 畫出 motifs ===
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

for i, ax in enumerate(axs.flatten()[:num_motifs]):
    motif_matrix = filters[:, :, i]

    # 轉成 DataFrame，對應 A,C,G,T 四個鹼基
    df = pd.DataFrame(motif_matrix, columns=['A', 'C', 'G', 'T'])

    # 使用 logomaker 畫圖
    logomaker.Logo(df, ax=ax)
    ax.set_title(f"Motif #{i+1}", fontsize=12)
    ax.set_xlabel("Position")
    ax.set_ylabel("Weight")

plt.tight_layout()
plt.savefig("./data/cattle/Rumen/motifs_identified.png", dpi=300)
plt.show()

print("✅ Motif visualization saved to: ./data/cattle/Rumen/motifs_identified.png")
