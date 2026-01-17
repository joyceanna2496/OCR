# model_multi.py
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# ---- 原本的 DeepOCR CNN branch ----
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


# ---- Multi-Input Model: CNN + local(16) + kmer(64) ----
def MultiInputDeepOCR():
    seq_input = Input(shape=(1000,4), name="seq_input")
    local_input = Input(shape=(16,), name="local_input")
    kmer_input  = Input(shape=(64,), name="kmer_input")

    # CNN branch
    cnn_out = DeepOCR_CNN(seq_input)

    # Concatenate all features
    merged = concatenate([cnn_out, local_input, kmer_input], name="concat_features")

    dense = Dense(300, activation='relu')(merged)
    dense = Dropout(0.6)(dense)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[seq_input, local_input, kmer_input], outputs=output)
    return model
