import numpy as np
from reseau_neuronal import Reseau_Neuronal
from extraction_mots import extraction_donnees_entree, division_donnees
from enregistrer_parametres_reseau import enregistrer_parametres
from extraction_caracteristiques import caracteristiques
import matplotlib.pyplot as plt
from time import time

# 330 mots par image par convention
nb_mots = 330

donnees = ['Antoine', 'Felix', 'Quentin', 'Raphael', 'Brayan']

donnees_entree = extraction_donnees_entree(donnees)
shape = donnees_entree.shape[1:]
donnees_entree = donnees_entree[:, shape[0]//2-15:shape[0]//2+15, shape[1]//2-45:shape[1]//2+45]
shape = donnees_entree.shape[1:]
caracteristiques = caracteristiques(donnees_entree)
del donnees_entree
donnees_entrainement, etiquettes_entrainement, donnees_test, etiquettes_test = division_donnees(caracteristiques,
                                                                                                nb_mots, 0.65)

Modele = Reseau_Neuronal([2, 3, 3, etiquettes_entrainement.shape[1]])
res1 = []
res2 = []
t1 = time()
for k in range(5):
    for i in range(100):
        Modele.Entrainement(donnees_entrainement, etiquettes_entrainement, vitesse_apprentissage=0.015-k*0.001)
        r1 = Modele.Test(donnees_entrainement, etiquettes_entrainement)
        r2 = Modele.Test(donnees_test, etiquettes_test)
        res1.append(r1)
        res2.append(r2)
print(time() - t1)
plt.plot(res1)
plt.plot(res2)
plt.show()
enregistrer_parametres(Modele, "Modele_caracteristiques2")
