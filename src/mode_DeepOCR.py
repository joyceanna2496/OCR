import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def mode_DeepOCR():

    def DW_block(X, filters, kernels):
        out = SeparableConv1D(filters, kernels, padding='same')(X)
        out = ReLU()(out)

        shortcut = Conv1D(filters, 1, padding='same')(X)
        out = Add()([shortcut, out])
        out = ReLU()(out)

        out = MaxPool1D(pool_size=3, strides=3)(out)
        out = Dropout(0.5)(out)
        return out

    inputs = Input((1000, 4))

    x = Conv1D(300, 19, padding="same")(inputs)
    x = DW_block(x, 200, 11)
    x = DW_block(x, 100, 9)

    x = GlobalAveragePooling1D()(x)

    x = Dense(300, activation='relu')(x)
    x = Dropout(0.6)(x)

    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs, output)