# preprocess2.py (final, flexible version)
import argparse
import numpy as np
import os
from itertools import product

# ----------------------------------
# FASTA Reader
# ----------------------------------
def read_fasta(path):
    seqs = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line.strip()
                buf = []
            else:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs

# ----------------------------------
# One-hot
# ----------------------------------
def onehot_base(c):
    if c == 'A': return [1,0,0,0]
    if c == 'C': return [0,1,0,0]
    if c == 'G': return [0,0,1,0]
    if c == 'T': return [0,0,0,1]
    return [0,0,0,0]  # N

def seq_to_onehot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        arr[i] = onehot_base(c)
    return arr

# ----------------------------------
# Modes: center / first1000 / window / smooth / segment
# ----------------------------------
def center_crop(seq, L):
    if len(seq) >= L:
        start = (len(seq) - L) // 2
        return seq[start:start+L]
    return seq + "N"*(L-len(seq))


def first_1000(seq, L):
    return seq[:L].upper().ljust(L, "N")


def sliding_windows(seq, window, step):
    out = []
    n = len(seq)
    if n <= window:
        out.append(center_crop(seq, window))
        return out
    for i in range(0, n-window+1, step):
        out.append(seq[i:i+window])
    return out


def smooth_fast(seq, length):
    seq = seq.upper()
    A=seq.count("A"); C=seq.count("C"); G=seq.count("G"); T=seq.count("T")
    total = A+C+G+T
    if total == 0:
        return "N"*length

    p = np.array([A,C,G,T], dtype=float)
    p = p / p.sum()

    cnt = (p * length).astype(int)
    diff = length - cnt.sum()
    cnt[np.argmax(cnt)] += diff

    s = "A"*cnt[0] + "C"*cnt[1] + "G"*cnt[2] + "T"*cnt[3]
    return s[:length]


def segment_reconstruct(seq, window, segment):
    seq = seq.upper()
    L = len(seq)
    seg_len = max(1, L//segment)
    segs = [seq[i:i+seg_len] for i in range(0, L, seg_len)]

    AC_ratio = []
    for s in segs:
        A=s.count("A"); C=s.count("C"); G=s.count("G"); T=s.count("T")
        total = A+C+G+T
        AC_ratio.append((A+C)/total if total else 0.5)

    p = np.array(AC_ratio)
    p = p / p.sum()

    new = ""
    for _ in range(window):
        s = segs[np.random.choice(len(segs), p=p)]
        A=s.count("A"); C=s.count("C"); G=s.count("G"); T=s.count("T")
        total=A+C+G+T
        if total == 0:
            new+="N"
        else:
            probs = np.array([A,C,G,T])/total
            new += np.random.choice(["A","C","G","T"], p=probs)
    return new


# ----------------------------------
# Flexible Local Feature
# ----------------------------------
def local_feature(seq, seg_num):
    L = len(seq)
    seg_len = L // seg_num
    v = []
    for i in range(seg_num):
        part = seq[i*seg_len:(i+1)*seg_len]
        total = len(part)
        A=part.count("A")/total if total else 0
        C=part.count("C")/total if total else 0
        G=part.count("G")/total if total else 0
        T=part.count("T")/total if total else 0
        v += [A,C,G,T]
    return np.array(v, dtype=np.float32)


# ----------------------------------
# Flexible k-mer (k=3 or 4)
# ----------------------------------
def kmer_feature(seq, k):
    seq = seq.upper()
    bases = ["A","C","G","T"]
    all_k = ["".join(p) for p in product(bases, repeat=k)]
    table = {kmer:0 for kmer in all_k}

    for i in range(len(seq)-k+1):
        s = seq[i:i+k]
        if s in table:
            table[s]+=1

    total = sum(table.values())
    if total == 0:
        return np.zeros(len(all_k), dtype=np.float32)
    return np.array([table[k]/total for k in all_k], dtype=np.float32)


# ----------------------------------
# Main
# ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True)
    parser.add_argument("--neg", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--mode", default="center",
        choices=["center","first1000","window","segment","smooth"])

    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--segment", type=int, default=10)

    parser.add_argument("--local_seg", type=int, choices=[0,4,5], default=0)
    parser.add_argument("--k", type=int, choices=[0,3,4], default=0)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Reading FASTA ...")
    pos = read_fasta(args.pos)
    neg = read_fasta(args.neg)

    X = []
    L_all = []
    K_all = []
    Y = []

    print(f"=== MODE: {args.mode} ===")
    print(f"Local seg: {args.local_seg}")
    print(f"k-mer: {args.k}")

    def process_one(seq):
        # choose mode
        if args.mode == "center":
            return center_crop(seq, args.window)

        elif args.mode == "first1000":
            return first_1000(seq, args.window)

        elif args.mode == "window":
            return sliding_windows(seq, args.window, args.step)

        elif args.mode == "segment":
            return segment_reconstruct(seq, args.window, args.segment)

        elif args.mode == "smooth":
            return smooth_fast(seq, args.window)

    # --------------------------
    # Iterate all sequences
    # --------------------------
    def add_sample(fixed_seq, label):
        X.append(seq_to_onehot(fixed_seq))
        if args.local_seg > 0:
            L_all.append(local_feature(fixed_seq, args.local_seg))
        if args.k > 0:
            K_all.append(kmer_feature(fixed_seq, args.k))
        Y.append(label)

    # positive
    for _, seq in pos.items():
        out_seq = process_one(seq)
        if isinstance(out_seq, list): # window mode
            for s in out_seq:
                add_sample(s, 1)
        else:
            add_sample(out_seq, 1)

    # negative
    for _, seq in neg.items():
        out_seq = process_one(seq)
        if isinstance(out_seq, list):
            for s in out_seq:
                add_sample(s, 0)
        else:
            add_sample(out_seq, 0)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32).reshape(-1,1)

    print("Saving CNN:", X.shape)
    np.save(os.path.join(args.out, "data_onehot.npy"), X)
    np.save(os.path.join(args.out, "label.npy"), Y)

    if args.local_seg > 0:
        L_all = np.array(L_all, dtype=np.float32)
        name = f"local_seg{args.local_seg}.npy"
        print("Saving LOCAL:", L_all.shape, "→", name)
        np.save(os.path.join(args.out, name), L_all)

    if args.k > 0:
        K_all = np.array(K_all, dtype=np.float32)
        name = f"kmer_k{args.k}.npy"
        print("Saving KMER:", K_all.shape, "→", name)
        np.save(os.path.join(args.out, name), K_all)

    print("\n🎉 DONE!")


if __name__ == "__main__":
    main()