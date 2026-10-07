# érosion : flou gaussien puis multiplier par 255/max(image) puis en-dessous d'un certain seuil
# on coupe verticalement, on établit une figure matplotlib des images itération après itération
import numpy as np
from pretraitement_images import flou_gaussien, flou_gaussien_vertical, regularisee
import matplotlib.pyplot as plt
from image28x28 import charger_image, image_en_28x28
from enregistrer_parametres_reseau import charger_parametres

N = charger_parametres("Lettres\ReconnaissanceLettres")

def separation(image):
    im = np.copy(image)

    deb = 0
    while sum(im[:, deb]) < 300:
        deb += 1

    fin = 27
    while sum(im[:, fin]) < 300:
        fin -= 1

    coupee = False
    while not coupee:
        im = regularisee(flou_gaussien_vertical(im))
        im[im<40] = 0.
        for i in range(deb, fin):
            if np.sum(im[:, i]) < 50:
                coupee = True
                coupure = i
        plt.imshow(im, cmap='gray')
        #plt.show()
    return deb+coupure

image = regularisee(charger_image("Lettres\Images_Lettres/image_m.png"))
s = separation(image)
im1, im2 = image[:, :s], image[:, s:]
im1 = image_en_28x28(im1, noir_sur_blanc=False, redresser_couleur=False)
im2 = image_en_28x28(im2, noir_sur_blanc=False, redresser_couleur=False)
