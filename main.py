# ====================================
# PROJET CASSIOPEE - Main
# ====================================

from src.preprocessing import chargement_donnees, aligner_et_filtrer_dataset, scaler_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.models import autoencoder_v4
from src.evaluation import sanity_check_dataset, verifier_visuellement, evaluer_ia_seule_multiclasse_hybride, evaluer_ia_seule_multiclasse_ae
from src.simulation import simuler_mass_shifts
import os

# ====================================
# 0. PARAMETRES GLOBAUX
# ====================================

EPOCHS = 50
BATCH_SIZE = 8

# ====================================
# 1. PREPARATION DES DONNEES
# ====================================

print("Chargement et préparation des données en cours...")

# 1. Chargement des données et sauvegarde en .npy dasn data/processed ( tenseur numpy 3D ) pour éviter de recharger les données à chaque fois 

cache_file = "data/processed/reference_spectrum_cache.npy"

if Path(cache_file).exists():
    print("⚡ Chargement ultra-rapide depuis le cache NumPy...")
    ref_a = np.load(cache_file)
else:
    print("Première lecture des CSV (ça va prendre du temps)...")
    ref_a = chargement_donnees("data/data/reference_spectrum")
    
    np.save(cache_file, ref_a)
    print("Cache sauvegardé !")

cache_file = "data/processed/reference_spectrum2_cache.npy"

if Path(cache_file).exists():
    print("⚡ Chargement ultra-rapide depuis le cache NumPy...")
    ref_b = np.load(cache_file)
else:
    print("Première lecture des CSV (ça va prendre du temps)...")
    ref_b = chargement_donnees("data/data2/reference spectrum")
    
    np.save(cache_file, ref_b)
    print("Cache sauvegardé !")

cache_file = "data/processed/others_cache.npy"

if Path(cache_file).exists():
    print("⚡ Chargement ultra-rapide depuis le cache NumPy...")
    others_brut = np.load(cache_file)
else:
    print("Première lecture des CSV (ça va prendre du temps)...")
    others_brut = chargement_donnees("data/data/other")
   
    np.save(cache_file, others_brut)
    print("Cache sauvegardé !")

cache_file = "data/processed/low_intensity_cache.npy"

if Path(cache_file).exists():
    print("⚡ Chargement ultra-rapide depuis le cache NumPy...")
    low_a = np.load(cache_file)
else:
    print("Première lecture des CSV (ça va prendre du temps)...")
    low_a = chargement_donnees("data/data/low_intensity")
    
    np.save(cache_file, low_a)
    print("Cache sauvegardé !")

cache_file = "data/processed/low_intensity2_cache.npy"

if Path(cache_file).exists():
    print("⚡ Chargement ultra-rapide depuis le cache NumPy...")
    low_b = np.load(cache_file)
else:
    print("Première lecture des CSV (ça va prendre du temps)...")
    low_b = chargement_donnees("data/data2/low intensity")
    
    np.save(cache_file, low_b)
    print("Cache sauvegardé !")

cache_file = "data/processed/mass_shift_cache.npy"

if Path(cache_file).exists():
    print("⚡ Chargement ultra-rapide depuis le cache NumPy...")
    mass_shift_brut = np.load(cache_file)
else:
    print("Première lecture des CSV (ça va prendre du temps)...")
    mass_shift_brut = chargement_donnees("data/data/mass_shift")
    
    np.save(cache_file, mass_shift_brut)
    print("Cache sauvegardé !")


ref_brut = np.concatenate([ref_a, ref_b], axis=0)
low_int_brut = np.concatenate([low_a, low_b], axis=0)

mz_maitre = ref_brut[0, :, 0]    

# 2. Séparation Train/Test (sur les normaux)

X_train_brut, X_test_norm_brut = train_test_split(ref_brut, test_size=0.2, random_state=42)

# 3. Alignement et Filtrage (On utilise le mz du train comme maître et on sélectionne seulement les fenêtres d'intérêt)

fichier_fenetres = "config/fenetres.txt"

X_train_filtre = aligner_et_filtrer_dataset(X_train_brut, mz_maitre, fichier_fenetres)
X_test_norm_filtre = aligner_et_filtrer_dataset(X_test_norm_brut, mz_maitre, fichier_fenetres)
low_int_filtre = aligner_et_filtrer_dataset(low_int_brut, mz_maitre, fichier_fenetres)
mass_shift_filtre = aligner_et_filtrer_dataset(mass_shift_brut, mz_maitre, fichier_fenetres)
others_filtre = aligner_et_filtrer_dataset(others_brut, mz_maitre, fichier_fenetres)

# 4. Scaling global

X_train_scaled, (X_test_norm_scaled, low_int_scaled, mass_shift_scaled, others_scaled), scaler = scaler_datasets(
    X_train_filtre, 
    X_test_norm_filtre, low_int_filtre, mass_shift_filtre, others_filtre
)

# 5. Simulation de Mass Shifts supplémentaires 
X_fake_mass_shift = simuler_mass_shifts(X_test_norm_scaled, severite_min=0.2, severite_max=0.8)
mass_shift_combined = np.concatenate((mass_shift_scaled, X_fake_mass_shift), axis=0)

print("Shape des données d'entrainement :", X_train_scaled.shape)

"""
# 6. Verification des données prétraitées

sanity_check_dataset("Train (Reference)", X_train_scaled)
sanity_check_dataset("Test (Reference)", X_test_norm_scaled)
sanity_check_dataset("Anomalies (Low Int)", low_int_scaled)
sanity_check_dataset("Anomalies (Mass Shift)", mass_shift_combined)
sanity_check_dataset("Anomalies (Others)", others_scaled)

# Test Visuel 

verifier_visuellement(X_train_scaled, "Train Normal")
verifier_visuellement(mass_shift_combined, "Mass Shift")
"""

# ====================================
# 2. ENTRAINEMENT DE L'AUTO-ENCODEUR
# ====================================

# 1. Création du modèle 
shape_entree = (X_train_scaled.shape[1], X_train_scaled.shape[2])
modele_ae = autoencoder_v4(input_shape=shape_entree, output_channels=1)

chemin_poids = "models_saved/ae_v4_weights.h5"

# 2. Entrainement (sauvegarde des poids du modèle dans un fichier h5 pour éviter de réentrainer le modèle à chaque fois)

if os.path.exists(chemin_poids):
    print("⚡ Poids trouvés ! Chargement instantané du modèle...")
    modele_ae.load_weights(chemin_poids)
else:
    print("Aucun poids trouvé. Lancement de l'entraînement...")
    os.makedirs("models_saved", exist_ok=True) 
    
    history = modele_ae.fit(
        X_train_scaled, 
        X_train_scaled[:, :, 1:], 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_norm_scaled, X_test_norm_scaled[:, :, 1:]),
        verbose=1
    )
    modele_ae.save_weights(chemin_poids)
    print("Poids sauvegardés avec succès !")


# ====================================
# 3. EVALUATION
# ====================================

print("Évaluation du modèle en cours...")

evaluer_ia_seule_multiclasse_hybride(X_test_norm_scaled, mass_shift_combined, low_int_scaled, modele_ae, seuil_mse=0.0001, seuil_int=310)
evaluer_ia_seule_multiclasse_ae(X_test_norm_scaled, mass_shift_combined, low_int_scaled, modele_ae, seuil_mse=0.0001)