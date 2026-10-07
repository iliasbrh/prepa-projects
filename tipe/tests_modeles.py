from enregistrer_parametres_reseau import charger_parametres
from reseau_neuronal import *
from extraction_mots import extraction_donnees_entree
from extraction_caracteristiques import caracteristiques
import matplotlib.pyplot as plt

modele_pixels = charger_parametres("Modele_pixels")
modele_caracteristiques = charger_parametres("Modele_caracteristiques")

nb_mots = 330
proportion_entrainement = 0.65
images_test = int(nb_mots*(1-proportion_entrainement))

donnees = ['Antoine', 'Felix', 'Quentin', 'Raphael', 'Brayan']

donnees_entree = extraction_donnees_entree(donnees)
shape = donnees_entree.shape[1:]
donnees_entree = donnees_entree[:, shape[0]//2-15:shape[0]//2+15, shape[1]//2-45:shape[1]//2+45]
shape = donnees_entree.shape[1:]
caracteristiques = caracteristiques(donnees_entree)

for k in range(1650):
    donnees_entree[k] = donnees_entree[k] / np.max(donnees_entree[k])

def resultats_de_chacun():
    print("Score Antoine : ")
    reponse = np.zeros((images_test, 5))
    reponse[0, :] = 1
    print("Modèle sur les pixels :", round(modele_pixels.Test(donnees_entree[330-images_test:330].reshape(images_test, 2700), reponse), 2))
    print("Modèle sur les caractéristiques :", round(modele_caracteristiques.Test(caracteristiques[330-images_test:330], reponse), 2))
    print("Score Felix : ")
    reponse = np.zeros((images_test, 5))
    reponse[1, :] = 1
    print("Modèle sur les pixels :", round(modele_pixels.Test(donnees_entree[660-images_test:660].reshape(images_test, 2700), reponse), 2))
    print("Modèle sur les caractéristiques :", round(modele_caracteristiques.Test(caracteristiques[660-images_test:660], reponse), 2))
    print("Score Quentin : ")
    reponse = np.zeros((images_test, 5))
    reponse[2, :] = 1
    print("Modèle sur les pixels :", round(modele_pixels.Test(donnees_entree[990-images_test:990].reshape(images_test, 2700), reponse), 2))
    print("Modèle sur les caractéristiques :", round(modele_caracteristiques.Test(caracteristiques[990-images_test:990], reponse), 2))
    print("Score Raphael : ")
    reponse = np.zeros((images_test, 5))
    reponse[3, :] = 1
    print("Modèle sur les pixels :", round(modele_pixels.Test(donnees_entree[1320-images_test:1320].reshape(images_test, 2700), reponse), 2))
    print("Modèle sur les caractéristiques :", round(modele_caracteristiques.Test(caracteristiques[1320-images_test:1320], reponse), 2))
    print("Score Brayan : ")
    reponse = np.zeros((images_test, 5))
    reponse[4, :] = 1
    print("Modèle sur les pixels :", round(modele_pixels.Test(donnees_entree[1650-images_test:1650].reshape(images_test, 2700), reponse), 2))
    print("Modèle sur les caractéristiques :", round(modele_caracteristiques.Test(caracteristiques[1650-images_test:1650], reponse), 2))

def cadrillage():
    plt.xlabel("Hauteur")
    plt.ylabel("Cursivité")
    couleurs = ["red", "green", "blue", "yellow", "pink"]
    N = 50
    for i in range(N+1):
        for j in range(N+1):
            plt.scatter(i/N, j/N, color=couleurs[np.argmax(modele_caracteristiques.Prediction(np.array([i/N, j/N]))[-1][-1])])
    plt.show()

def tests_par_k_car(k, n):
    score = 0
    for i in range(n):
        cobaye = np.random.randint(0, 5)
        print(cobaye)
        images = [np.random.randint((cobaye+1)*330-images_test, (cobaye+1)*330) for l in range(k)]
        predictions = [np.argmax(modele_caracteristiques.Prediction(caracteristiques[im])[-1][-1]) for im in images]
        decompte = [predictions.count(l) for l in range(5)]
        indmax = 0
        for j in range(1, 5):
            if decompte[j] > decompte[indmax]:
                indmax = j
        if cobaye == indmax:
            score += 1
    print(score/n)
    return score/n
def tests_par_k_pixel(k, n):
    score = 0
    for i in range(n):
        cobaye = np.random.randint(0, 5)
        images = [np.random.randint((cobaye+1)*330-images_test, (cobaye+1)*330) for l in range(k)]
        predictions = [np.argmax(modele_pixels.Prediction(donnees_entree[im].reshape(2700))[-1][-1]) for im in images]
        decompte = [predictions.count(l) for l in range(5)]
        indmax = 0
        for j in range(1, 5):
            if decompte[j] > decompte[indmax]:
                indmax = j
        if cobaye == indmax:
            score += 1
    return score/n
resultats_de_chacun()
#tests_par_k(50, 500)
#cadrillage()
n = 40
L1 = [0 for i in range(n)]
L2 = [0 for i in range(n)]
for i in range(n):
    L1[i] = tests_par_k_car(i+1, 500)
    L2[i] = tests_par_k_pixel(i+1, 500)
    print(L1, L2)
    print(i)
plt.plot(L1, color='red')
plt.plot(L2 , color='green')
plt.show()