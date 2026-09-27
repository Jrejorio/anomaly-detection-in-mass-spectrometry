"""
Module de Préparation des Données 
Contient les fonctions pour charger, aligner, filtrer et normaliser les spectres de masse.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def chargement_donnees(chemin):
    """
    Charge tous les fichiers CSV d'un dossier dans un tenseur numpy 3D.
    Applique le passage au log1p sur les intensités.
    
    Shape de sortie attendue : (nb_spectres, nb_points, 2)
    """
    chemin = Path(chemin)
    fichiers = list(chemin.glob("*.csv"))
    
    if not fichiers:
        raise ValueError(f"Aucun fichier CSV trouvé dans le dossier {chemin}")

    # On lit le premier fichier pour avoir la dimension de référence
    df_test = pd.read_csv(fichiers[0])
    nb_lignes = len(df_test)
    nb_colonnes = len(df_test.columns)

    fichiers_valides = [f for f in fichiers if pd.read_csv(f, nrows=1).shape[1] == nb_colonnes]
    
    '''
    for fichier in fichiers:
        df = pd.read_csv(fichier)
        if df.values.shape == (nb_lignes, nb_colonnes):
            fichiers_valides.append(fichier)
    '''

    nb_fichiers = len(fichiers_valides)
    d1 = np.zeros((nb_fichiers, nb_lignes))
    d2 = np.zeros((nb_fichiers, nb_lignes))

    print(f"Chargement du dossier : {chemin.name}")

    for i, fichier in enumerate(tqdm(fichiers_valides, desc="Lecture des CSV", unit="fichier")):
        try:
            df = pd.read_csv(fichier)
            d1[i] = df['mz'].values
            # Passage au log des intensités
            d2[i] = np.log1p(df['spectrum'].values)
        except Exception as e:
            print(f"Erreur de lecture sur le fichier {fichier.name}: {e}")


    return np.stack((d1, d2), axis=-1)


def lire_fenetres(chemin_fenetres):
    """
    Lit le fichier texte contenant les fenêtres m/z d'intérêt.
    Retourne une liste de tuples (start, end).
    """
    fenetres = []
    with open(chemin_fenetres, 'r') as f:
        for line in f:
            line = line.strip()
            # On ignore les lignes mal formatées
            if ';' in line and not line.startswith('['):
                parts = line.split(';')
                try:
                    fenetres.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return fenetres


def aligner_et_filtrer_dataset(dataset_3d, mz_reference, chemin_fenetres):
    """
    1. Aligne l'intensité de chaque spectre sur l'axe mz_reference via interpolation.
    2. Applique le filtrage pour ne garder que les points dans les fenêtres.
    
    Retourne un tenseur de taille fixe (N_spectres, N_points_filtres, 2).
    """
    nb_spectres = dataset_3d.shape[0]
    nb_points_ref = len(mz_reference)
    
    
    data_alignee = np.zeros((nb_spectres, nb_points_ref, 2))
    data_alignee[:, :, 0] = mz_reference 
    
    for i in range(nb_spectres):
        current_mz = dataset_3d[i, :, 0]
        current_int = dataset_3d[i, :, 1]
        
        f_interp = interp1d(current_mz, current_int, bounds_error=False, fill_value=0)
        data_alignee[i, :, 1] = f_interp(mz_reference)

    
    fenetres = lire_fenetres(chemin_fenetres)
    masque = np.zeros(nb_points_ref, dtype=bool)
    
    for start, end in fenetres:
        masque |= (mz_reference >= start) & (mz_reference <= end)

    data_finale = data_alignee[:, masque, :]
    return data_finale


def scaler_datasets(X_train, *autres_datasets):
    """
    Applique un MinMaxScaler sur le jeu d'entraînement, puis transforme les autres jeux.
    Gère automatiquement le reshape 3D -> 2D -> 3D requis par scikit-learn.
    """
    scaler = MinMaxScaler()
    
    
    X_train_2d = X_train.reshape(-1, 2)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    
    
    datasets_scaled = []
    for dataset in autres_datasets:
        dataset_2d = dataset.reshape(-1, 2)
        dataset_scaled = scaler.transform(dataset_2d).reshape(dataset.shape)
        datasets_scaled.append(dataset_scaled)
        
    return X_train_scaled, datasets_scaled, scaler