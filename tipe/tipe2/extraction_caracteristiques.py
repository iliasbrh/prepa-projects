import numpy as np
from extraction_mots import extraction_donnees_entree
import matplotlib.pyplot as plt
from random import choice

def Sigmoide(x: np.ndarray)->np.ndarray:
    return 1 / (1 + (np.exp(-x)))

def caracteristiques(donnees):
    N = donnees.shape[0]
    shape = donnees.shape[1:]
    deb = np.zeros(N)
    fin = np.zeros(N)
    for i in range(N):
        s = np.max(donnees[i][3:-3, 3:-3], axis=0)
        s = s > 0.2
        deb[i] = np.argmax(s) + 3
        fin[i] = shape[1] - np.argmax(s[::-1]) - 3
    deb = deb.astype(int)
    fin = fin.astype(int)

    hauteurs = np.zeros(N)
    for i in range(N):
        tmp = []
        for j in range(deb[i], fin[i]):
            s = donnees[i][:, j]
            s = s > 0.2
            tmp.append((shape[0]-np.argmax(s[::-1]))-np.argmax(s))
        hauteurs[i] = np.mean(tmp) / (shape[0]-6)
    hauteurs = hauteurs / max(hauteurs)
    hauteurs = Sigmoide((hauteurs - 0.4)*8)
    facteur = max(hauteurs) - min(hauteurs)
    hauteurs = (hauteurs - min(hauteurs)) / facteur
    hauteurs **= 1/2

    cursivite = np.zeros(N)
    for i in range(N):
        visite = []
        pixels_visites = {(n, m):False for n in range(shape[0]) for m in range(shape[1])}

        def profondeur(pixel):
            visite[-1].append(pixel)
            pixels_visites[pixel] = True
            voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for v in voisins:
                voisin = (pixel[0] + v[0], pixel[1] + v[1])
                if 0 <= voisin[0] < shape[0] and 0 <= voisin[1] < shape[1]:
                    if donnees[i][voisin] > 0.1 and not pixels_visites[voisin]:
                        profondeur(voisin)

        for j in range(shape[0]):
            for k in range(shape[1]):
                if donnees[i][j, k] > 0.1 and not pixels_visites[(j, k)]:
                    visite.append([])
                    profondeur((j, k))

        for L in visite:
            if len(L) > 25:
                cursivite[i] += 1
        cursivite[i] -= 0.95

        cursivite[i] /= (fin[i] - deb[i])
    cursivite /= np.max(cursivite)
    cursivite **= 1/2

    return np.concatenate((hauteurs.reshape(N, 1), cursivite.reshape(N, 1)), axis=1)

# pas à montrer dans l'annexe
def test():
    donnees = ['Antoine', 'Quentin', 'Felix', 'Raphael', 'Brayan']

    donnees_entree = extraction_donnees_entree(donnees)

    carac = caracteristiques(donnees_entree)
    hauteur, cursivite = carac[:, 0], carac[:, 1]

    for i in range(100):
        ind1, ind2, ind3, ind4, ind5 = np.random.randint(0, 329), np.random.randint(330, 660), np.random.randint(660, 990), np.random.randint(990, 1320), np.random.randint(1320, 1650)
        ind = choice([ind1, ind2, ind3, ind4, ind5])
        print("hauteur : ", hauteur[ind], "   cursivite : ", cursivite[ind], "\n")
        plt.imshow(donnees_entree[ind], cmap='gray')
        plt.show()

def caracteristiques_de_chacun():
    donnees = ['Antoine', 'Quentin', 'Felix', 'Raphael', 'Brayan']

    donnees_entree = extraction_donnees_entree(donnees)

    carac = caracteristiques(donnees_entree)
    hauteur, cursivite = carac[:, 0], carac[:, 1]

    print("Antoine :\nhauteur = {}, {} et cursivite = {}, {}".format(np.mean(hauteur[:330]), np.std(hauteur[:330]), np.mean(cursivite[:330]), np.std(cursivite[:330])))
    print("Quentin :\nhauteur = {}, {} et cursivite = {}, {}".format(np.mean(hauteur[330:660]), np.std(hauteur[330:660]), np.mean(cursivite[330:660]), np.std(cursivite[330:660])))
    print("Félix :\nhauteur = {}, {} et cursivite = {},  {}".format(np.mean(hauteur[660:990]), np.std(hauteur[660:990]), np.mean(cursivite[660:990]), np.std(cursivite[660:990])))
    print("Raphael :\nhauteur = {}, {} et cursivite = {}, {}".format(np.mean(hauteur[990:1320]), np.std(hauteur[990:1320]), np.mean(cursivite[990:1320]), np.std(cursivite[990:1320])))
    print("Brayan :\nhauteur = {}, {} et cursivite = {}, {}".format(np.mean(hauteur[1320:]), np.std(hauteur[1320:]), np.mean(cursivite[1320:]), np.std(cursivite[1320:])))

caracteristiques_de_chacun()