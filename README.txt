Exécuter le main.py pour voir les résultats s'afficher:

main.py appelle d'autres fichiers src :
i. src/preprocessing.py permet de transformer les données brutes en tenseur numpy 3D (nb spectres, nb points, 2) et les sauvegarde dans des fichiers .npy 
(data/preprocessed) pour éviter de tout charger à chaque fois (c'est long). Il permet aussi de scaler les données pour que ce soit plus simple à digérer dans 
l'autoencodeur.
ii. src.simulation permet de simuler des mass_shifts pour avoir plus de données.
iii. src/models.py code l'architecture de l'autoenodeur. Les poids sont sauvegardés dans des fichiers .h5 (models_saved) pour éviter l'entrainement à chaque fois 
(même si avec la version finale l'entrainement et rapide)
iv. src/evaluation.py donne des métriques pour évaluer la performance du modèle et affiche des graphiques

=> au final des bons résultats trouvés 

