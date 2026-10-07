from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def charger_image(f):
    # Renvoie l'image du fichier f sous forme de tableau numpy
    return np.array(Image.open(f).convert('L'))

def image_en_28x28(img, noir_sur_blanc=True, redresser_couleur=True):
    # Renvoie l'image en taille 28x28
    hauteur, longueur = img.shape

    if longueur < 28:
        return epaissir(img)

    if redresser_couleur:
        img[img>130] = 255


    res = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            debutH, finH = int(i*hauteur/28), int((i+1)*hauteur/28)
            debutL, finL = int(j*longueur/28), int((j+1)*longueur/28)
            res[i, j] = np.mean(img[debutH:finH, debutL:finL])

    if noir_sur_blanc:
        res = 255 - res

    return res / 255

def epaissir(image):
    l = np.shape(image)[1]
    res = np.zeros((28, 28))
    res[:, 14-(l//2):14-(l//2)+l] = image
    return res