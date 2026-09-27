"""
Module des Modèles -
Contient les architectures de réseaux de neurones (Auto-encodeurs) 
et les fonctions de perte.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

def weighted_mse(y_true, y_pred):
    """
    Fonction de perte personnalisée : Mean Squared Error pondérée.
    Pénalise davantage les erreurs de reconstruction sur les hauts pics d'intensité.
    """
    poids = y_true + 1.0 
    return K.mean(K.square(y_true - y_pred) * poids)

def autoencoder_v4(input_shape=(740, 2), output_channels=1):
    """
    Auto-encodeur optimisé pour les spectres filtrés (740 points).
    """
    inputs = layers.Input(shape=input_shape)
    

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding="same", activation="relu")(inputs)
    
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    

    encoded = layers.Conv1D(4, kernel_size=3, padding="same", activation="relu")(x)
    
    
    x = layers.Conv1DTranspose(32, kernel_size=5, strides=2, padding="same", activation="relu")(encoded)
    
    x = layers.Conv1DTranspose(16, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    
    
    decoded = layers.Conv1D(output_channels, kernel_size=3, padding="same", activation="sigmoid")(x)
    
    model = models.Model(inputs, decoded, name="Cassiopee_AE_v4")
    
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

def build_and_summary():
    
    model = autoencoder_v4()
    model.summary()
    return model