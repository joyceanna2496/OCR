import argparse
import numpy as np
import os
from itertools import product

# ============================
# FASTA 讀取
# ============================
def read_fasta(path):
    seqs = {}
    name = None
    buff = []

    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buff)
                name = line.strip()
                buff = []
            else:
                buff.append(line.strip())

        if name:
            seqs[name] = "".join(buff)

    return seqs


# ============================
# One-hot encode
# ============================
def nuc_to_onehot(c):
    c = c.upper()
    if c == "A": return [1,0,0,0]
    if c == "C": return [0,1,0,0]
    if c == "G": return [0,0,1,0]
    if c == "T": return [0,0,0,1]
    return [0,0,0,0]  # N


def seq_to_matrix(seq):
    arr = np.zeros((len(seq), 4), dtype=np.int8)
    for i, c in enumerate(seq):
        arr[i] = nuc_to_onehot(c)
    return arr


# ============================
# Mode: center
# ============================
def center_crop(seq, L=1000):
    seq = seq.upper()
    if len(seq) >= L:
        start = (len(seq) - L) // 2
        return seq[start:start + L]
    return seq + "N" * (L - len(seq))


# ============================
# Mode: first 1000
# ============================
def first_1000(seq, L=1000):
    seq = seq.upper()
    if len(seq) >= L:
        return seq[:L]
    return seq + "N" * (L - len(seq))


# ============================
# Mode: sliding windows
# ============================
def sliding_windows(seq, window=1000, step=200):
    seq = seq.upper()
    n = len(seq)
    if n <= window:
        return [center_crop(seq, window)]

    out = []
    for i in range(0, n - window + 1, step):
        out.append(seq[i:i+window])

    return out


# ============================
# Mode: segment sampling
# ============================
def segment_reconstruct(seq, window=1000, segment=10):
    seq = seq.upper()
    L = len(seq)
    seg_len = max(1, L // segment)
    segs = [seq[i:i+seg_len] for i in range(0, L, seg_len)]

    # 計算各區段 (A+C) 比例
    weights = []
    for s in segs:
        A = s.count("A"); C = s.count("C")
        G = s.count("G"); T = s.count("T")
        total = A + C + G + T
        weights.append((A + C) / total if total else 0.25)

    weights = np.array(weights)
    weights = weights / weights.sum()

    new_seq = ""
    for _ in range(window):
        seg = segs[np.random.choice(len(segs), p=weights)]
        A = seg.count("A"); C = seg.count("C")
        G = seg.count("G"); T = seg.count("T")
        total = A+C+G+T
        if total == 0:
            new_seq += "N"
        else:
            probs = np.array([A,C,G,T]) / total
            new_seq += np.random.choice(["A","C","G","T"], p=probs)

    return new_seq


# ============================
# Mode: smooth (global composition)
# ============================
def smooth_fast(seq, L=1000):
    seq = seq.upper()
    A = seq.count("A"); C = seq.count("C")
    G = seq.count("G"); T = seq.count("T")
    total = A+C+G+T

    if total == 0:
        return "N"*L

    p = np.array([A, C, G, T], dtype=float) / total

    counts = (p * L).astype(int)
    diff = L - counts.sum()
    if diff != 0:
        counts[np.argmax(p)] += diff

    blocks = ["A"*counts[0], "C"*counts[1], "G"*counts[2], "T"*counts[3]]
    seq_out = "".join(blocks)
    return seq_out[:L]


# ============================
# Local 16 feature
# ============================
def local_feature(seq):
    seq = seq.upper()
    L = len(seq)
    seg = L // 4

    feat = []
    for i in range(4):
        part = seq[i*seg:(i+1)*seg]
        total = len(part)
        if total == 0:
            feat += [0,0,0,0]
        else:
            A = part.count("A")/total
            C = part.count("C")/total
            G = part.count("G")/total
            T = part.count("T")/total
            feat += [A,C,G,T]

    return np.array(feat, dtype=np.float32)


# ============================
# k-mer 64 維
# ============================
def kmer_feature(seq, k=3):
    seq = seq.upper()

    bases = ["A","C","G","T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    count = {k:0 for k in kmers}

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in count:
            count[kmer] += 1

    total = sum(count.values())
    if total == 0:
        return np.zeros(len(kmers), dtype=np.float32)

    return np.array([count[k] / total for k in kmers], dtype=np.float32)


# ============================
# Main
# ============================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True)
    parser.add_argument("--neg", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--mode",
        choices=["center","first1000","window","segment","smooth"],
        default="center"
    )

    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--segment", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("讀取正樣本:", args.pos)
    pos = read_fasta(args.pos)

    print("讀取負樣本:", args.neg)
    neg = read_fasta(args.neg)

    X = []
    L_feat = []
    K_feat = []
    Y = []

    print(f"\n=== 使用模式: {args.mode} ===\n")

    # ---------------------------
    # 處理資料
    # ---------------------------
    def process_one(seq, label):

        # 決定 fixed 序列們 (可能多個 window)
        if args.mode == "center":
            seqs = [center_crop(seq, args.window)]

        elif args.mode == "first1000":
            seqs = [first_1000(seq, args.window)]

        elif args.mode == "window":
            seqs = sliding_windows(seq, args.window, args.step)

        elif args.mode == "segment":
            seqs = [segment_reconstruct(seq, args.window, args.segment)]

        elif args.mode == "smooth":
            seqs = [smooth_fast(seq, args.window)]

        # 轉成 feature
        for s in seqs:
            X.append(seq_to_matrix(s))
            L_feat.append(local_feature(s))
            K_feat.append(kmer_feature(s))
            Y.append(label)

    # 正樣本
    for name, seq in pos.items():
        process_one(seq, 1)

    # 負樣本
    for name, seq in neg.items():
        process_one(seq, 0)

    # 轉 numpy
    X = np.array(X)
    L_feat = np.array(L_feat)
    K_feat = np.array(K_feat)
    Y = np.array(Y)

    # 輸出
    print("CNN shape :", X.shape)
    print("Local shape:", L_feat.shape)
    print("K-mer shape:", K_feat.shape)
    print("Label shape:", Y.shape)

    np.save(os.path.join(args.out, "data_onehot.npy"), X)
    np.save(os.path.join(args.out, "local.npy"), L_feat)
    np.save(os.path.join(args.out, "kmer.npy"), K_feat)
    np.save(os.path.join(args.out, "label.npy"), Y)

    print("\n🎉 完成 Hybrid 預處理！")
    print("✔ data_onehot.npy")
    print("✔ local.npy")
    print("✔ kmer.npy")
    print("✔ label.npy")


if __name__ == "__main__":
    main()