# train_multi.py
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_multi import MultiInputDeepOCR
import os

# enable GPU dynamic memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("✅ GPU enabled")

def evaluate(model, x, local, kmer, y):
    pred_prob = model.predict([x, local, kmer])
    pred = (pred_prob > 0.5).astype(int)

    auc = roc_auc_score(y, pred_prob)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    Se = tp / (tp + fn)
    Sp = tn / (tn + fp)
    f1 = f1_score(y, pred)

    MCC = ((tp*tn)-(fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    return acc, auc, Se, Sp, MCC, f1


def train(out, seq, local, kmer, label, val_split, random_split):

    model = MultiInputDeepOCR()
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])

    print(model.summary())

    train_x, test_x, train_l, test_l, train_k, test_k, train_y, test_y = train_test_split(
        seq, local, kmer, label, test_size=random_split
    )

    ckpt = ModelCheckpoint(out + "best_model.keras", save_best_only=True, monitor='val_loss', verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=12, verbose=1)

    model.fit(
        [train_x, train_l, train_k],
        train_y,
        batch_size=32,
        epochs=12,
        validation_split=val_split,
        callbacks=[ckpt, early],
        shuffle=True
    )

    best = tf.keras.models.load_model(out + "best_model.keras", compile=False)
    best.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return evaluate(best, test_x, test_l, test_k, test_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--local", required=True)
    parser.add_argument("--kmer", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--val", type=float, required=True)
    parser.add_argument("--random", type=float, required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data ...")
    seq = np.load(args.seq).astype("float32")
    local = np.load(args.local).astype("float32")
    kmer = np.load(args.kmer).astype("float32")
    label = np.load(args.label).reshape(-1,1).astype("float32")

    acc, auc, Se, Sp, MCC, f1 = train(
        args.out, seq, local, kmer, label, args.val, args.random
    )

    print("\n=== FINAL RESULTS ===")
    print("Accuracy :", acc)
    print("AUC      :", auc)
    print("Se       :", Se)
    print("Sp       :", Sp)
    print("MCC      :", MCC)
    print("F1       :", f1)


if __name__ == "__main__":
    main()