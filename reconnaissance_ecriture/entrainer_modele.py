from charger_base_de_donnees import charger_n_images
from reseau_neuronal import Reseau_Neuronal
from enregistrer_parametres_reseau import enregistrer_parametres
from pretraitement_images import pretraitement
import matplotlib.pyplot as plt
import numpy as np

# Utilisation de la base de données Emnist
images_entrainement = "emnist-letters-train-images-idx3-ubyte.bin"
etiquettes_entrainement = "emnist-letters-train-labels-idx1-ubyte.bin"
n_entrainement = 124800 # 124800 images dans la base de données d'entraînement

images_test = "emnist-letters-test-images-idx3-ubyte.bin"
etiquettes_test = "emnist-letters-test-labels-idx1-ubyte.bin"
n_test = 20800 # 20800 images dans la base de données de test

# Tuples contenant à la fois les images (tableau de forme (n, 784)) et les étiquettes (de forme n), renvoyés par charger_n_images
donnees_entrainement = charger_n_images(images_entrainement, etiquettes_entrainement, n_entrainement)
donnees_test = charger_n_images(images_test, etiquettes_test, n_test)

# On met les étiquettes sous la forme de tableaux de 0 et un 1 pour l'utilisation dans la fonction Entrainement de réseau neuronal
sorties_attendues_entrainement = np.zeros((n_entrainement, 26))
for i in range(n_entrainement):
    # On soustrait 1 car les lettres sont numérotées à partir de 1 au lieu de 0,
    # et on les veut à partir de 0 par soucis de clarté
    sorties_attendues_entrainement[i][int(donnees_entrainement[1][i]) - 1] = 1
sorties_attendues_test = np.zeros((n_test, 26))
for i in range(n_test):
    sorties_attendues_test[i][int(donnees_test[1][i]) - 1] = 1

Reconnaissance_Lettres = Reseau_Neuronal([784, 70, 50, 26])

# Découpage des données pour faire plusieurs tests au fur et à mesure de l'apprentissage
iterations = 50
tranche_entrainement, tranche_test = n_entrainement // iterations, n_test // iterations
nb_repetitions = 4 # nombre de fois où les données sont montrées au réseau neuronal
print("{} itérations vont être effectuées, à {} reprises.".format(iterations, nb_repetitions))

echantillon_images_test = pretraitement(donnees_test[0], aleatoire=True)

scores = []
for i in range(nb_repetitions):
    for j in range(iterations):
        debut, fin = j * tranche_entrainement, (j + 1) * tranche_entrainement
        echantillon_images_entrainement = pretraitement(donnees_entrainement[0][debut:fin], aleatoire=True)
        Reconnaissance_Lettres.Entrainement(echantillon_images_entrainement, sorties_attendues_entrainement[debut:fin],
                                    vitesse_apprentissage=0.1)

        score = Reconnaissance_Lettres.Test(echantillon_images_test, sorties_attendues_test)

        print("Nombre de répétitions : {}. Score après {} itérations : {}%".format(i+1, j+1, round(score*100, 1)))
        scores.append(round(score*100, 1))

print("Fin de l'entraînement")

# Affichage de l'évolution du score
plt.plot(np.linspace(0, len(scores), len(scores)), scores)
plt.show()

enregistrer_parametres(Reconnaissance_Lettres, "ReconnaissanceLettres")