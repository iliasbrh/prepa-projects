import numpy as np
from scipy.ndimage import zoom, rotate

# Pour élargir la base de données, on prend les données et on leur applique des rotations et zooms aléatoires
# pour qu'à chaque fois qu'une image est remontrée elle soit différente
def zoom_et_rotation(image, facteur_zoom, angle_rotation):
    # Fonction faite par chatgpt
    assert image.shape == (28, 28), "L'image doit être de taille 28x28"
    # Zoom de l'image
    image_zoomee = zoom(image, facteur_zoom)
    # Rotation de l'image
    image_apres_rotation = rotate(image_zoomee, angle_rotation)

    # Redimensionnement de l'image pour qu'elle conserve sa taille d'origine (28x28)
    if image_apres_rotation.shape[0] < 28: # si l'image est trop petite on la complete de 0 (pixels noirs) et on la centre
        biais = (28 - image_apres_rotation.shape[0]) // 2
        image_redimensionnee = np.zeros((28, 28))
        image_redimensionnee[biais:image_apres_rotation.shape[0]+biais, biais:image_apres_rotation.shape[0]+biais] = image_apres_rotation
    else: #  si l'image est trop grande on ne conserve que le centre
        ligne_debut = (image_apres_rotation.shape[0] - 28) // 2
        colonne_debut = (image_apres_rotation.shape[1] - 28) // 2
        image_redimensionnee = image_apres_rotation[ligne_debut:ligne_debut + 28, colonne_debut:colonne_debut + 28]
    return image_redimensionnee

def pretraitement(images, aleatoire=True):
    n = images.shape[0]
    if aleatoire:
        for i in range(n):
            # zoom et rotation aléatoires de l'image
            images[i] = zoom_et_rotation(images[i].reshape(28, 28), 0.9, 0).reshape(784)
    images[images<0.001] = 0.
    return images / 255 # on souhaite des données entre 0 et 1 pour le réseau neuronal

def regularisee(image):
    tmp = image - np.min(image)
    return tmp * (255 / np.max(tmp))

def flou_gaussien(image, repetitions=1):
    for i in range(repetitions):
        s = np.shape(image)
        im = np.zeros(s)
        for i in range(s[0]):
            for j in range(s[1]):
                im[i, j] = np.mean(image[max(i-1, 0):min(i+1, s[0]), max(0, j-1):min(s[1], j+1)])
        image = regularisee(im)
    return image

def flou_gaussien_vertical(image, repetitions=1):
    for i in range(repetitions):
        s = np.shape(image)
        im = np.zeros(s)
        for i in range(s[0]):
            for j in range(s[1]):
                im[i, j] = min(np.mean(image[max(i-1, 0):min(i+1, s[0]), j]), image[i, j])
        image = np.copy(im)
    return image