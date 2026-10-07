import os
from reseau_neuronal import Reseau_Neuronal
import numpy as np

def enregistrer_parametres(N: Reseau_Neuronal, chemin: str):
    assert not os.path.exists(chemin), "Ce chemin d'accès existe déjà"

    # Les parametres sont stockees comme un dossier contenant deux sous-dossiers (poids et biais)
    # eux-mêmes contenant les tableaux numpy des paramètres numérotés dans l'ordre
    os.makedirs(chemin + "\poids")
    for i in range(len(N.poids)):
        np.savetxt(chemin + "\poids\poids" + str(i + 1), N.poids[i])

    os.makedirs(chemin + "\\biais")
    for i in range(len(N.biais)):
        np.savetxt(chemin + "\\biais\\biais" + str(i + 1), N.biais[i])


def charger_parametres(chemin):
    assert os.path.exists(chemin), "Ce chemin d'accès n'existe pas"

    poids, biais = [], []

    i = 1
    # Extraction des poids du réseau jusqu'à ce qu'il n'y en ait plus
    while True:
        direction = chemin + "\poids\poids" + str(i)
        if not os.path.exists(direction):
            break
        else:
            poids.append(np.loadtxt(direction))
        i += 1

    i = 1
    # Extraction des biais
    while True:
        direction = chemin + "\\biais\\biais" + str(i)
        if not os.path.exists(direction):
            break
        else:
            biais.append(np.loadtxt(direction))
        i += 1

    couches = [np.shape(poids[0])[0]]
    for i in biais:
        couches.append(np.size(i))

    Reseau = Reseau_Neuronal(couches)
    Reseau.poids = poids
    Reseau.biais = biais

    return Reseau
