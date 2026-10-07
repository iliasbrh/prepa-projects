from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def extraction_png(image):
    taille = image.shape
    debut_horizontal, debut_vertical, fin_horizontal, fin_vertical = 0, 0, taille[0] - 1, taille[1] - 1

    while np.mean(image[debut_horizontal, :]) > 250:
        debut_horizontal += 1
    while np.mean(image[:, debut_vertical]) > 250:
        debut_vertical += 1
    while np.mean(image[fin_horizontal, :]) > 250:
        fin_horizontal -= 1
    while np.mean(image[:, fin_vertical]) > 250:
        fin_vertical -= 1
    im = image[debut_horizontal:fin_horizontal, debut_vertical:fin_vertical]

    def decoupage_lignes(image):
        taille = image.shape
        lignes = []
        l = 0
        while l < taille[0] and len(lignes) <= 33:
            shift = 0
            while l+shift < taille[0] and np.mean(image[l+shift]) > 125:
                shift += 1
            if shift > 1:
                lignes.append(image[l+3:l+shift-2])
                l += shift
            l += 1
        return lignes

    def decoupage_colonnes(image):
        taille = image.shape
        colonnes = []
        c = 0
        while c < taille[1] and len(colonnes) <= 10:
            shift = 0
            while c+shift < taille[1] and np.mean(image[:, c+shift:c+shift+2]) > 60:
                shift += 1
            if shift > 1:
                colonnes.append(image[:, c+3:c+shift-3])
                c += shift
            c += 1
        return colonnes
    mots = []
    for l in decoupage_lignes(im):
        mots.extend(decoupage_colonnes(l))
    return mots

def uniformisation_taille(mots):
    largeur, longueur = 0, 0
    for l in mots:
        largeur, longueur = max(largeur, l.shape[0]), max(longueur, l.shape[1])
    for i in range(len(mots)):
        largeur_mot, longueur_mot = np.shape(mots[i])
        res = np.full((largeur, longueur), 255)
        res[largeur//2 - (largeur_mot // 2):largeur // 2 - (largeur_mot // 2) + largeur_mot,
            longueur//2 - (longueur_mot // 2):longueur // 2 - (longueur_mot // 2) + longueur_mot] = mots[i]
        res[res>235] = 255
        mots[i] = res
    for i in range(len(mots)):
        # Pour éviter que des traits du découpage ne restent sur les bords de l'image
        mots[i][:4, :] = 255
        mots[i][:, :4] = 255
        mots[i][-2:, :] = 255
        mots[i][:, -4:] = 255
    return mots

def extraction_donnees_entree(fichiers):
    mots = []
    for f in fichiers:
        mots.extend(extraction_png(np.array(Image.open("cobayes\\" + f + ".png"))[:, :, 0]))
    mots = uniformisation_taille(mots)
    mots_array = np.zeros((len(mots), np.shape(mots[0])[0], np.shape(mots[0])[1]))
    for i in range(len(mots)):
        mots_array[i, :, :] = mots[i]
    return 1 - (mots_array/255)

def division_donnees(donnees, nb_mots, proportion_entrainement, vertical=False):
    nb_images_entrainement = int(proportion_entrainement * nb_mots)
    nb_images_test = 330 - nb_images_entrainement

    if vertical:
        donnees = donnees.reshape((donnees.shape[0], donnees.shape[1]*donnees.shape[2]))

    taille_donnees = donnees.shape[0]
    nombre_auteurs = taille_donnees // 330

    donnees_entrainement = np.zeros((nombre_auteurs * nb_images_entrainement, donnees.shape[1]))
    donnees_test = np.zeros((nombre_auteurs * nb_images_test, donnees.shape[1]))

    etiquettes_entrainement = np.zeros((nombre_auteurs * nb_images_entrainement, nombre_auteurs))
    etiquettes_test = np.zeros((nombre_auteurs * nb_images_test, nombre_auteurs))

    for i in range(nombre_auteurs):
        donnees_entrainement[i * nb_images_entrainement:(i + 1) * nb_images_entrainement] = donnees[i * 330:i * 330 +
                                                                                                nb_images_entrainement]
        donnees_test[i * nb_images_test:(i + 1) * nb_images_test] = donnees[i * 330 + nb_images_entrainement:(i + 1) * 330]
        etiquettes_entrainement[i * nb_images_entrainement:(i + 1) * nb_images_entrainement, i] = 1
        etiquettes_test[i * nb_images_test:(i + 1) * nb_images_test, i] = 1

    # Mélange aléatoire de l'ordre des données
    entrainement_melange = np.zeros(donnees_entrainement.shape)
    entrainement_etiquettes_melange = np.zeros(etiquettes_entrainement.shape)
    permutation = np.random.permutation(donnees_entrainement.shape[0])
    for ancien_indice, nouvel_indice in enumerate(permutation):
        entrainement_melange[nouvel_indice] = donnees_entrainement[ancien_indice]
        entrainement_etiquettes_melange[nouvel_indice] = etiquettes_entrainement[ancien_indice]

    return entrainement_melange, entrainement_etiquettes_melange, donnees_test, etiquettes_test

# tests visuels pas à mettre dans l'annexe
# mots_maman = extraction_donnees_entree(['maman.png'])
# mots_papa = extraction_donnees_entree(['papa.png'])
# print(np.shape(mots_maman), np.shape(mots_papa))
def afficher_mots_sur_une_page():
    nb_mots = 330

    donnees = ['Antoine', 'Felix', 'Quentin', 'Raphael', 'Brayan']

    donnees_entree = extraction_donnees_entree(donnees)
    shape = donnees_entree.shape[1:]
    donnees_entree = donnees_entree[:, shape[0]//2-15:shape[0]//2+15, shape[1]//2-45:shape[1]//2+45]

    for j in range(len(donnees)):
        fig = plt.figure(figsize=(33, 10))
        for i in range(nb_mots):
            fig.add_subplot(33, 10, i+1)
            plt.imshow(donnees_entree[nb_mots*j+i])
        plt.show()

def afficher_mots():
    nb_mots = 330
    donnees = ['Antoine', 'Felix', 'Quentin', 'Raphael', 'Brayan']

    donnees_entree = extraction_donnees_entree(donnees)
    shape = donnees_entree.shape[1:]
    donnees_entree = donnees_entree[:, shape[0] // 2 - 15:shape[0] // 2 + 15, shape[1] // 2 - 45:shape[1] // 2 + 45]
    while True:
        plt.imshow(donnees_entree[np.random.randint(0, nb_mots*len(donnees))], cmap='gray')
        plt.show()



