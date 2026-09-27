import numpy as np

def simuler_mass_shifts(spectres_normaux, severite_min=0.2, severite_max=0.8):
    """
    Prend un bloc de spectres normaux et applique un décalage m/z (Mass Shift).
    Retourne un bloc de la même taille contenant les anomalies artificielles.
    """
    print(f"Simulation de Mass Shifts (décalage entre {severite_min} et {severite_max} m/z)...")
    
    
    spectres_simules = np.copy(spectres_normaux)
    nb_spectres = spectres_simules.shape[0]
    
    for i in range(nb_spectres):
        
        shift = np.random.uniform(severite_min, severite_max)
        
        direction = np.random.choice([-1, 1])
        
    
        spectres_simules[i, :, 0] += (shift * direction)
        
    print(f"{nb_spectres} anomalies de type Mass Shift générées !")
    return spectres_simules