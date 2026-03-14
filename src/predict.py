import argparse
import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ========== DNA to ONE-HOT ==========
def change_char_label(seq, pn):
    bed_dict = {}
    out = open(pn + ".txt", "w")
    for line in open(seq, "r"):
        if line[0] == ">":
            key = line.strip()
            bed_dict[key] = []
        else:
            bed_dict[key].append(line.strip())

    for k, v in bed_dict.items():
        out.write("".join(v) + "\n")
    out.close()


def onehot(filename, nuc_number=1000):
    lines = open(filename).read().splitlines()
    num = len(lines)

    data = np.zeros((num, nuc_number, 4), dtype=np.int32)

    base_map = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0]
    }

    for i, seq in enumerate(lines):
        seq = seq.strip().upper()
        for j in range(min(len(seq), nuc_number)):
            data[i, j, :] = base_map.get(seq[j], [0, 0, 0, 0])

    return data


# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, required=True, help="Model file (.keras)")
    parser.add_argument("--seq", type=str, required=True, help="FASTA sequences")
    args = parser.parse_args()

    out = args.out
    modelf = args.model
    seq = args.seq

    if not os.path.exists(out):
        os.makedirs(out)

    # Convert fasta → pure sequence lines
    tmp_txt = os.path.join(out, "pos_1hot")
    change_char_label(seq, pn=tmp_txt)

    # Load & one-hot encode
    data = onehot(tmp_txt + ".txt")
    os.remove(tmp_txt + ".txt")

    # Load full model (.keras format)
    model = load_model(modelf, compile=False)

    # Prediction
    pred_rate = model.predict(data)
    pred_binary = (pred_rate > 0.5).astype(int)

    # Extract headers
    headers = [line.strip() for line in open(seq) if line.startswith(">")]

    df = pd.DataFrame({"ID": headers, "Pred": pred_binary.flatten()})
    df.to_csv(out + "/pred.csv", index=False, header=False)

    print("Prediction saved to:", out + "/pred.csv")


if __name__ == "__main__":
    main()