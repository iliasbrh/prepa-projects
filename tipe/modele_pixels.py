import numpy as np
from reseau_neuronal import Reseau_Neuronal
import matplotlib.pyplot as plt
from extraction_mots import extraction_donnees_entree, division_donnees
from enregistrer_parametres_reseau import enregistrer_parametres
from time import time

# 330 mots par image par convention
nb_mots = 330

donnees = ['Antoine', 'Felix', 'Quentin', 'Raphael', 'Brayan']

donnees_entree = extraction_donnees_entree(donnees)
shape = donnees_entree.shape[1:]
donnees_entree = donnees_entree[:, shape[0]//2-15:shape[0]//2+15, shape[1]//2-45:shape[1]//2+45]
shape = donnees_entree.shape[1:]

for k in range(nb_mots*len(donnees)):
    donnees_entree[k] = donnees_entree[k] / np.max(donnees_entree[k])

donnees_entrainement, etiquettes_entrainement, donnees_test, etiquettes_test = division_donnees(donnees_entree, nb_mots,
                                                                                                0.65, vertical=True)

Modele = Reseau_Neuronal([donnees_entrainement.shape[1], 16, 16, etiquettes_entrainement.shape[1]])
res1 = []
res2 = []
t1 = time()
for i in range(50):
    Modele.Entrainement(donnees_entrainement, etiquettes_entrainement, vitesse_apprentissage=0.03)
    r1 = Modele.Test(donnees_entrainement, etiquettes_entrainement)
    r2 = Modele.Test(donnees_test, etiquettes_test)
    res1.append(r1)
    res2.append(r2)
print(time() - t1)
plt.plot(res1)
plt.plot(res2)
plt.show()

enregistrer_parametres(Modele, "Modele_pixels")
