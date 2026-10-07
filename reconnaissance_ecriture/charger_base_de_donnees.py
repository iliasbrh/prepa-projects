import numpy as np

def charger_n_images(chemin_acces_donnees_image, chemin_acces_donnees_etiquettes, n_images):
    # Code donné sur internet avec la base de données
    images = np.zeros((n_images, 784)) # 784 car on stocke des images de taille 28x28
    etiquettes = np.zeros(n_images)

    etiquettes_fichier = open(chemin_acces_donnees_etiquettes, "rb")
    images_fichier = open(chemin_acces_donnees_image, "rb")

    # On passe les informations en tête de fichier
    images_fichier.read(16)
    etiquettes_fichier.read(8)

    # Lecture des données
    for i in range(n_images):
        etiquettes[i] = ord(etiquettes_fichier.read(1))
        for j in range(784):
            images[i, j] = ord(images_fichier.read(1))
        print("Extraction des données : {}%".format(round(100*i/n_images, 1)))
    print("Données extraites avec succès")

    etiquettes_fichier.close()
    images_fichier.close()

    # Mélange des données, car initialement les images de la base de données sont stockées dans l'ordre alphabétique
    indices_aleatoires = np.random.rand(n_images).argsort()
    for i in range(784):
        images[:, i] = np.take_along_axis(images[:, i], indices_aleatoires, axis=0)
    etiquettes = np.take_along_axis(etiquettes, indices_aleatoires, axis=0)
    return images, etiquettes
