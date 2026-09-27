"""
Module d'Évaluation 
Contient les fonctions pour tester le modèle, calculer les scores (MSE, Intensité)
et afficher les rapports de performance : ROC, P-R, et Matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                             roc_curve, roc_auc_score, precision_recall_curve, average_precision_score)

def sanity_check_dataset(nom, dataset):
    print(f"--- Diagnostic : {nom} ---")
    
    # Vérification de la Shape
    print(f"Shape attendue  : (N, 740, 2)")
    print(f"Shape actuelle  : {dataset.shape}")
    if dataset.shape[1:] != (740, 2):
        print("ERREUR CRITIQUE : La dimension des points ou colonnes est fausse !")
        
    # Vérification des NaNs et Infs
    nb_nans = np.isnan(dataset).sum()
    nb_infs = np.isinf(dataset).sum()
    if nb_nans > 0 or nb_infs > 0:
        print(f"ERREUR CRITIQUE : Contient {nb_nans} NaNs et {nb_infs} Infs !")
    else:
        print("Données propres (0 NaN, 0 Inf)")
        
    # Vérification du Scaling (sur la colonne intensité : index 1)
    min_int = np.min(dataset[:, :, 1])
    max_int = np.max(dataset[:, :, 1])
    print(f"Intensité Min   : {min_int:.4f}")
    print(f"Intensité Max   : {max_int:.4f}")
    print("\n")

def verifier_visuellement(dataset, nom="Spectre Aléatoire"):

    idx = np.random.randint(0, dataset.shape[0])
    spectre = dataset[idx]
    
    plt.figure(figsize=(10, 4))
    plt.plot(spectre[:, 0], spectre[:, 1], color='blue')
    plt.title(f"Test Visuel : {nom} (Index {idx})")
    plt.xlabel("m/z")
    plt.ylabel("Intensité (Normalisée)")
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluer_ia_seule_multiclasse_hybride(X_norm, X_mass, X_low, modele_ae, seuil_mse=0.0004, seuil_int=310):
    """
    Génère un tableau de bord à 6 graphiques pour analyser le comportement 
    de l'IA face aux données Saines, Mass Shifts et Low Intensity.
    """
    print("Calcul des prédictions (MSE et Intensité) en cours...")
    
    # 1. Calcul des MSE (Erreur de forme)
    mse_norm = np.mean(np.square(X_norm[:, :, 1:] - modele_ae.predict(X_norm, verbose=0)), axis=(1, 2))
    mse_mass = np.mean(np.square(X_mass[:, :, 1:] - modele_ae.predict(X_mass, verbose=0)), axis=(1, 2))
    mse_low = np.mean(np.square(X_low[:, :, 1:] - modele_ae.predict(X_low, verbose=0)), axis=(1, 2))
    
    # 2. Calcul des Intensités totales (Volume)
    int_norm = np.sum(X_norm[:, :, 1], axis=1)
    int_mass = np.sum(X_mass[:, :, 1], axis=1)
    int_low = np.sum(X_low[:, :, 1], axis=1)
    
    # 3. Regroupement pour les métriques globales
    y_true = np.array([0] * len(mse_norm) + [1] * (len(mse_mass) + len(mse_low)))
    tous_mses = np.concatenate([mse_norm, mse_mass, mse_low])
    tous_ints = np.concatenate([int_norm, int_mass, int_low])
    y_pred = ((tous_mses > seuil_mse) | (tous_ints < seuil_int)).astype(int)
    
    print(f"\n{'='*50}")
    print(f" RÉSULTATS GLOBAUX DU MODELE HYBRIDE (Seuil MSE = {seuil_mse}, Seuil Intensité = {seuil_int})")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalie"]))

    
    # CRÉATION DU DASHBOARD (2 lignes, 3 colonnes)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Tableau de Bord d'Évaluation Multiclasse - Modèle hybride", fontsize=16, fontweight='bold')

    # 1. Histogramme des MSE (En haut à gauche)
    axes[0, 0].hist(mse_norm, bins=30, alpha=0.6, color='blue', label='Saines')
    axes[0, 0].hist(mse_low, bins=30, alpha=0.6, color='red', label='Low Intensity')
    axes[0, 0].hist(mse_mass, bins=30, alpha=0.7, color='green', label='Mass Shift')
    axes[0, 0].axvline(seuil_mse, color='black', lw=2, linestyle='--', label='Seuil MSE')
    axes[0, 0].set_title("1. Distribution des Erreurs (MSE)")
    axes[0, 0].set_xlabel("MSE (Échelle Log)")
    axes[0, 0].set_ylabel("Nombre")
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    #2. Scatter Plot : Intensité vs MSE (En haut au centre) 

    axes[0, 1].scatter(int_norm, mse_norm, color='blue', alpha=0.5, label='Saines', s=15)
    axes[0, 1].scatter(int_low, mse_low, color='red', alpha=0.7, label='Low Intensity', s=15)
    axes[0, 1].scatter(int_mass, mse_mass, color='green', alpha=0.7, label='Mass Shift', s=15)
    axes[0, 1].axhline(seuil_mse, color='black', linestyle='--', label='Filtre IA (MSE)')
    axes[0, 1].axvline(310, color='grey', linestyle='-.', label='Filtre Intensité (310)') # Remplace 310 par ton vrai seuil
    axes[0, 1].set_title("2. Scatter Plot (Intensité vs MSE)")
    axes[0, 1].set_xlabel("Intensité Totale")
    axes[0, 1].set_ylabel("Erreur MSE (Échelle Log)")
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Boxplot des MSE (En haut à droite)
    axes[0, 2].boxplot([mse_norm, mse_low, mse_mass], labels=['Saines', 'Low Int.', 'Mass Shift'])
    axes[0, 2].axhline(seuil_mse, color='black', linestyle='--')
    axes[0, 2].set_title("3. Boxplot des Erreurs (MSE)")
    axes[0, 2].set_ylabel("MSE (Échelle Log)")
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    #  4. Courbes ROC Comparatives (En bas à gauche) 
    # Étape A : Normalisation du MSE (0 à 1, où 1 = Forte Erreur / Anomalie)
    mse_min, mse_max = np.min(tous_mses), np.max(tous_mses)
    score_mse_norm = (tous_mses - mse_min) / (mse_max - mse_min + 1e-10)
    # Étape B : Normalisation de l'Intensité (On inverse, car Basse Intensité = Anomalie)
    int_invers = -tous_ints 
    int_min, int_max = np.min(int_invers), np.max(int_invers)
    score_int_norm = (int_invers - int_min) / (int_max - int_min + 1e-10)
    # Étape C : Le Score Hybride (Le maximum des deux probabilités d'anomalie)
    score_hybride = np.maximum(score_mse_norm, score_int_norm)
    # Calcul des 3 courbes
    fpr_ia, tpr_ia, _ = roc_curve(y_true, score_mse_norm)
    auc_ia = roc_auc_score(y_true, score_mse_norm)
    fpr_int, tpr_int, _ = roc_curve(y_true, score_int_norm)
    auc_int = roc_auc_score(y_true, score_int_norm)
    fpr_hyb, tpr_hyb, _ = roc_curve(y_true, score_hybride)
    auc_hyb = roc_auc_score(y_true, score_hybride)
    # Tracé des 3 courbes
    axes[1, 0].plot(fpr_ia, tpr_ia, color='green', lw=2, alpha=0.7, label=f'IA (AUC = {auc_ia:.2f})')
    axes[1, 0].plot(fpr_int, tpr_int, color='red', lw=2, alpha=0.7, label=f'Intensité (AUC = {auc_int:.2f})')
    # La courbe hybride en bleu plus épais pour la mettre en valeur
    axes[1, 0].plot(fpr_hyb, tpr_hyb, color='blue', lw=3, label=f'Hybride (AUC = {auc_hyb:.2f})')
    axes[1, 0].plot([0, 1], [0, 1], color='grey', linestyle='--')
    axes[1, 0].set_title("4. Comparatif des Courbes ROC")
    axes[1, 0].set_xlabel("Faux Positifs")
    axes[1, 0].set_ylabel("Vrais Positifs")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)

    #  5. Courbes Précision-Rappel Comparatives (En bas au centre) 

    # Calcul des courbes PR et de l'Average Precision (AP)
    prec_ia, rec_ia, _ = precision_recall_curve(y_true, score_mse_norm)
    ap_ia = average_precision_score(y_true, score_mse_norm)
    prec_int, rec_int, _ = precision_recall_curve(y_true, score_int_norm)
    ap_int = average_precision_score(y_true, score_int_norm)
    prec_hyb, rec_hyb, _ = precision_recall_curve(y_true, score_hybride)
    ap_hyb = average_precision_score(y_true, score_hybride)
    # Tracé des 3 courbes
    axes[1, 1].plot(rec_ia, prec_ia, color='green', lw=2, alpha=0.7, label=f'IA (AP = {ap_ia:.2f})')
    axes[1, 1].plot(rec_int, prec_int, color='red', lw=2, alpha=0.7, label=f'Intensité (AP = {ap_int:.2f})')
    # La courbe hybride en bleu plus épais pour la mettre en valeur
    axes[1, 1].plot(rec_hyb, prec_hyb, color='blue', lw=3, label=f'Hybride (AP = {ap_hyb:.2f})')
    axes[1, 1].set_title("5. Comparatif Précision-Rappel")
    axes[1, 1].set_xlabel("Rappel (Recall)")
    axes[1, 1].set_ylabel("Précision")
    # Pour la PR, la légende est souvent mieux en bas à gauche ou en haut à droite
    axes[1, 1].legend(loc="lower left") 
    axes[1, 1].grid(True, alpha=0.3)

    #  6. Matrice de Confusion (En bas à droite) 
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalie"])
    disp.plot(cmap=plt.cm.Blues, ax=axes[1, 2], colorbar=False)
    axes[1, 2].set_title("6. Matrice de Confusion (Modèle hybride)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.show()

def evaluer_ia_seule_multiclasse_ae(X_norm, X_mass, X_low, modele_ae, seuil_mse=0.0004, seuil_int=310):
    """
    Génère un tableau de bord à 6 graphiques pour analyser le comportement 
    de l'IA face aux données Saines, Mass Shifts et Low Intensity.
    """
    print("Calcul des prédictions (MSE et Intensité) en cours...")
    
    # 1. Calcul des MSE (Erreur de forme)
    mse_norm = np.mean(np.square(X_norm[:, :, 1:] - modele_ae.predict(X_norm, verbose=0)), axis=(1, 2))
    mse_mass = np.mean(np.square(X_mass[:, :, 1:] - modele_ae.predict(X_mass, verbose=0)), axis=(1, 2))
    mse_low = np.mean(np.square(X_low[:, :, 1:] - modele_ae.predict(X_low, verbose=0)), axis=(1, 2))
    
    # 2. Calcul des Intensités totales (Volume)
    int_norm = np.sum(X_norm[:, :, 1], axis=1)
    int_mass = np.sum(X_mass[:, :, 1], axis=1)
    int_low = np.sum(X_low[:, :, 1], axis=1)
    
    # 3. Regroupement pour les métriques globales
    y_true = np.array([0] * len(mse_norm) + [1] * (len(mse_mass) + len(mse_low)))
    tous_mses = np.concatenate([mse_norm, mse_mass, mse_low])
    y_pred = (tous_mses > seuil_mse).astype(int)
    
    print(f"\n{'='*50}")
    print(f" RÉSULTATS GLOBAUX DE L'IA SEULE (Seuil MSE = {seuil_mse})")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalie"]))

    
    # CRÉATION DU DASHBOARD (2 lignes, 3 colonnes)
  
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Tableau de Bord d'Évaluation Multiclasse - IA Seule", fontsize=16, fontweight='bold')

    # 1. Histogramme des MSE (En haut à gauche)
    axes[0, 0].hist(mse_norm, bins=30, alpha=0.6, color='blue', label='Saines')
    axes[0, 0].hist(mse_low, bins=30, alpha=0.6, color='red', label='Low Intensity')
    axes[0, 0].hist(mse_mass, bins=30, alpha=0.7, color='green', label='Mass Shift')
    axes[0, 0].axvline(seuil_mse, color='black', lw=2, linestyle='--', label='Seuil MSE')
    axes[0, 0].set_title("1. Distribution des Erreurs (MSE)")
    axes[0, 0].set_xlabel("MSE (Échelle Log)")
    axes[0, 0].set_ylabel("Nombre")
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    #  2. Scatter Plot : Intensité vs MSE (En haut au centre)
    
    axes[0, 1].scatter(int_norm, mse_norm, color='blue', alpha=0.5, label='Saines', s=15)
    axes[0, 1].scatter(int_low, mse_low, color='red', alpha=0.7, label='Low Intensity', s=15)
    axes[0, 1].scatter(int_mass, mse_mass, color='green', alpha=0.7, label='Mass Shift', s=15)
    axes[0, 1].axhline(seuil_mse, color='black', linestyle='--', label='Filtre IA (MSE)')
    axes[0, 1].axvline(310, color='grey', linestyle='-.', label='Filtre Intensité (310)') # Remplace 310 par ton vrai seuil
    axes[0, 1].set_title("2. Scatter Plot (Intensité vs MSE)")
    axes[0, 1].set_xlabel("Intensité Totale")
    axes[0, 1].set_ylabel("Erreur MSE (Échelle Log)")
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    #  3. Boxplot des MSE (En haut à droite) 
    axes[0, 2].boxplot([mse_norm, mse_low, mse_mass], labels=['Saines', 'Low Int.', 'Mass Shift'])
    axes[0, 2].axhline(seuil_mse, color='black', linestyle='--')
    axes[0, 2].set_title("3. Boxplot des Erreurs (MSE)")
    axes[0, 2].set_ylabel("MSE (Échelle Log)")
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Courbe ROC Globale (En bas à gauche)
    fpr, tpr, _ = roc_curve(y_true, tous_mses)
    auc_score = roc_auc_score(y_true, tous_mses)
    axes[1, 0].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.2f}')
    axes[1, 0].plot([0, 1], [0, 1], color='grey', linestyle='--')
    axes[1, 0].set_title("4. Courbe ROC Globale")
    axes[1, 0].set_xlabel("Faux Positifs")
    axes[1, 0].set_ylabel("Vrais Positifs")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Courbe Précision-Rappel (En bas au centre)
    precision, recall, _ = precision_recall_curve(y_true, tous_mses)
    axes[1, 1].plot(recall, precision, color='darkorange', lw=2)
    axes[1, 1].set_title("5. Courbe Précision-Rappel")
    axes[1, 1].set_xlabel("Rappel (Recall)")
    axes[1, 1].set_ylabel("Précision")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Matrice de Confusion (En bas à droite)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalie"])
    disp.plot(cmap=plt.cm.Blues, ax=axes[1, 2], colorbar=False)
    axes[1, 2].set_title("6. Matrice de Confusion (IA Seule)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.show()
    
    # 5. Matrice de Confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalie"])
    fig, ax = plt.subplots(figsize=(6, 4))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Matrice de Confusion Globale (IA Seule)")
    plt.show()