# train_multi.py
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_multi import build_flexible_model
import os

# enable GPU dynamic memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("✅ GPU enabled")


# ------------------------------
# Evaluation function
# ------------------------------
def evaluate(model, inputs, y_true):
    pred_prob = model.predict(inputs)
    pred = (pred_prob > 0.5).astype(int)

    auc = roc_auc_score(y_true, pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    Se = tp / (tp + fn) if (tp + fn) > 0 else 0
    Sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, pred)

    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    MCC = ((tp*tn)-(fp*fn)) / denom if denom != 0 else 0

    # ⭐ 四捨五入到小數點第 4 位
    return (
        round(acc, 4),
        round(auc, 4),
        round(Se, 4),
        round(Sp, 4),
        round(MCC, 4),
        round(f1, 4)
    )

# ------------------------------
# Training wrapper
# ------------------------------
def run_training(out, seq, local, kmer, label, val_split, random_split, fold):

    has_local = local is not None
    has_kmer  = kmer is not None

    local_dim = local.shape[1] if has_local else None
    kmer_dim  = kmer.shape[1] if has_kmer else None

    print("\n=== MODEL CONFIGURATION ===")
    print("Use CNN (seq): ✔")
    print(f"Use local     : {has_local}  (dim={local_dim})")
    print(f"Use k-mer     : {has_kmer}  (dim={kmer_dim})")
    print("===========================\n")

    # --------------------------
    # K-Fold
    # --------------------------
    if fold:
        print(f"▶ Running {fold}-Fold Cross Validation")

        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
        results = np.zeros(6)

        for i, (tr, te) in enumerate(skf.split(seq, label), 1):
            print(f"\n----- Fold {i} -----")

            X_tr, X_te = seq[tr], seq[te]
            y_tr, y_te = label[tr], label[te]

            inputs_train = [X_tr]
            inputs_test  = [X_te]

            if has_local:
                L_tr, L_te = local[tr], local[te]
                inputs_train.append(L_tr)
                inputs_test.append(L_te)

            if has_kmer:
                K_tr, K_te = kmer[tr], kmer[te]
                inputs_train.append(K_tr)
                inputs_test.append(K_te)

            model = build_flexible_model(has_local, local_dim, has_kmer, kmer_dim)
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(0.001),
                          metrics=['accuracy'])

            ckpt = ModelCheckpoint(out + f"best_fold_{i}.keras", save_best_only=True, monitor='val_loss')
            early = EarlyStopping(monitor='val_loss', patience=12)

            model.fit(
                inputs_train,
                y_tr,
                batch_size=32,
                epochs=12,
                validation_split=val_split,
                callbacks=[ckpt, early],
                shuffle=True,
                verbose=1
            )

            best = tf.keras.models.load_model(out + f"best_fold_{i}.keras", compile=False)
            best.compile(loss='binary_crossentropy', optimizer='adam')

            acc, auc, Se, Sp, MCC, f1 = evaluate(best, inputs_test, y_te)
            results += np.array([acc, auc, Se, Sp, MCC, f1])

        print("\n=== FINAL AVG RESULTS ===")
        print("ACC =", results[0]/fold)
        print("AUC =", results[1]/fold)
        print("Se  =", results[2]/fold)
        print("Sp  =", results[3]/fold)
        print("MCC =", results[4]/fold)
        print("F1  =", results[5]/fold)
        return


    # --------------------------
    # Random Train-Test Split
    # --------------------------
    print("▶ Running Random Split")

    splits = [seq]
    if has_local: splits.append(local)
    if has_kmer:  splits.append(kmer)
    splits.append(label)

    arrays = train_test_split(*splits, test_size=random_split, shuffle=True)
    idx = 0

    X_tr = arrays[idx]; idx+=1
    X_te = arrays[idx]; idx+=1

    inputs_train = [X_tr]
    inputs_test  = [X_te]

    if has_local:
        L_tr = arrays[idx]; idx+=1
        L_te = arrays[idx]; idx+=1
        inputs_train.append(L_tr)
        inputs_test.append(L_te)

    if has_kmer:
        K_tr = arrays[idx]; idx+=1
        K_te = arrays[idx]; idx+=1
        inputs_train.append(K_tr)
        inputs_test.append(K_te)

    y_tr = arrays[idx]; idx+=1
    y_te = arrays[idx]; idx+=1

    model = build_flexible_model(has_local, local_dim, has_kmer, kmer_dim)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])

    ckpt = ModelCheckpoint(out + "best_model.keras", save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=12)

    model.fit(
        inputs_train,
        y_tr,
        batch_size=32,
        epochs=10,
        validation_split=val_split,
        callbacks=[ckpt, early],
        shuffle=True,
        verbose=1
    )

    best = tf.keras.models.load_model(out + "best_model.keras", compile=False)
    best.compile(loss='binary_crossentropy', optimizer='adam')

    acc, auc, Se, Sp, MCC, f1 = evaluate(best, inputs_test, y_te)

    print("\n=== FINAL RESULTS ===")
    print("ACC =", acc)
    print("AUC =", auc)
    print("Se  =", Se)
    print("Sp  =", Sp)
    print("MCC =", MCC)
    print("F1  =", f1)


# ------------------------------
# main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--local")
    parser.add_argument("--kmer")
    parser.add_argument("--label", required=True)
    parser.add_argument("--val", type=float, required=True)
    parser.add_argument("--random", type=float)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    seq = np.load(args.seq).astype("float32")
    label = np.load(args.label).reshape(-1,1).astype("float32")

    local = np.load(args.local).astype("float32") if args.local else None
    kmer  = np.load(args.kmer).astype("float32") if args.kmer else None

    if (args.random is None and args.fold is None):
        print("❌ ERROR: use --random or --fold")
        return

    run_training(args.out, seq, local, kmer, label, args.val, args.random, args.fold)


if __name__ == "__main__":
    main()