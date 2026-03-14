# model_multi.py
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# -----------------------
# CNN branch (unchanged)
# -----------------------
def DeepOCR_CNN(seq_input):

    def DW_block(X, filters, kernels):
        out = SeparableConv1D(filters, kernels, depth_multiplier=1, padding='same')(X)
        out = ReLU()(out)
        X_c = Conv1D(filters, 1, padding='same')(X)
        out = add([X_c, out])
        out = ReLU()(out)
        out = MaxPooling1D(pool_size=3, strides=3)(out)
        out = Dropout(0.5)(out)
        return out

    x = Conv1D(filters=300, kernel_size=19, padding="same")(seq_input)
    x = DW_block(x, filters=200, kernels=11)
    x = DW_block(x, filters=100, kernels=9)
    x = GlobalAveragePooling1D()(x)

    return x


# -----------------------
# Multi-input flexible model
# -----------------------
def build_flexible_model(has_local, local_dim, has_kmer, kmer_dim):
    """
    自動建立輸入：
    - seq_input：一定要
    - local_input：如果 has_local == True 才加入
    - kmer_input：如果 has_kmer == True 才加入
    """

    # ---- seq CNN branch ----
    seq_input = Input(shape=(1000,4), name="seq_input")
    cnn_out = DeepOCR_CNN(seq_input)

    inputs = [seq_input]
    merged_features = [cnn_out]

    # ---- local optional ----
    if has_local:
        local_input = Input(shape=(local_dim,), name="local_input")
        inputs.append(local_input)
        merged_features.append(local_input)

    # ---- kmer optional ----
    if has_kmer:
        kmer_input = Input(shape=(kmer_dim,), name="kmer_input")
        inputs.append(kmer_input)
        merged_features.append(kmer_input)

    # ---- merge all ----
    if len(merged_features) > 1:
        merged = concatenate(merged_features, name="concat_features")
    else:
        merged = merged_features[0]

    dense = Dense(300, activation='relu')(merged)
    dense = Dropout(0.6)(dense)
    output = Dense(1, activation='sigmoid')(dense)

    return Model(inputs=inputs, outputs=output)