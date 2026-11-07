"""
Models module.

Neural network architecture of the autoencoder.
"""

from tensorflow.keras import layers, models


def autoencoder_v4(input_shape=(740, 2), output_channels=1):
    """
    Convolutional 1D autoencoder for the filtered spectra (740 points).

    Takes (m/z, intensity) as input and reconstructs the intensity only.
    The bottleneck compresses the spectrum 4x along the m/z axis, down to 4 channels.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    encoded = layers.Conv1D(4, kernel_size=3, padding="same", activation="relu")(x)

    # Decoder
    x = layers.Conv1DTranspose(32, kernel_size=5, strides=2, padding="same", activation="relu")(encoded)
    x = layers.Conv1DTranspose(16, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    decoded = layers.Conv1D(output_channels, kernel_size=3, padding="same", activation="sigmoid")(x)

    model = models.Model(inputs, decoded, name="Cassiopee_AE_v4")
    model.compile(optimizer="adam", loss="mse")

    return model
