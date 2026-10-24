# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv1D, Flatten, Dense, Dropout
from kymatio.keras import Scattering1D


# Extract scattering features from each channel before convolution.
class SpikingScattering(Layer):


    def __init__(self, J=2, Q=2, T=None, **kwargs):

        super().__init__(**kwargs)
        self.J = J
        self.Q = Q
        self.T = T

    def build(self, input_shape):
        self.T = input_shape[1]
        self.scattering = Scattering1D(J=self.J, Q=self.Q)
        self.conv = Conv1D(16, 3, strides=2)
        super().build(input_shape)

    def call(self, inputs):
        output_list = []
        # Process the two input channels independently.
        for channel in range(inputs.shape[-1]):
            single_channel = inputs[..., channel]
            S = self.scattering(single_channel)
            # Arrange coefficients as [batch, time, features] for Conv1D.
            S = tf.transpose(S, perm=[0, 2, 1])
            output_list.append(S)

        # Fuse channel features before the strided convolution.
        x = tf.concat(output_list, axis=-1)
        return self.conv(x)


def build_model(Len_signal, class_num):

    inputs = Input(shape=(Len_signal, 2))
    x = SpikingScattering(J=2, Q=2, T=Len_signal)(inputs)

    # Use dense layers to map scattering features to class probabilities.
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    outputs = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=[inputs], outputs=outputs)
    return model
