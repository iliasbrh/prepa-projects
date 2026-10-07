from enregistrer_parametres_reseau import charger_parametres
from charger_base_de_donnees import charger_n_images
from pretraitement_images import pretraitement, flou_gaussien
from image28x28 import *

N = charger_parametres("ReconnaissanceLettres")
print(N.couches)

def donnees_test():
    n = 20800 # 20800 images dans les données de test en tout

    donnees = charger_n_images("emnist-letters-test-images-idx3-ubyte.bin", "emnist-letters-test-labels-idx1-ubyte.bin", n_images=n)
    images = pretraitement(donnees[0], aleatoire=True)

    for i in range(n):
        print(chr(np.argmax(N.Prediction(images[i])[-1][-1]) + 97), chr(int(donnees[1][i] - 1) + 97))
        plt.imshow(np.transpose(images[i].reshape(28, 28)), cmap='gray')
        plt.show()

def images_exterieures():
    while True:
        test = input("Quelle lettre tester : ")
        im = flou_gaussien(np.transpose(image_en_28x28(charger_image("Images_Lettres/"+test+".png"))), repetitions=1)
        im = im.reshape(784)
        print("Prediction : "+chr(np.argmax(N.Prediction(im)[-1][-1]) + 97))
        plt.imshow(np.transpose(im.reshape(28, 28)), cmap='gray')
        # plt.show()

def reconnaissance_en_direct():
    import pygame, sys
    fenetre = pygame.display.set_mode((280, 280))
    fenetre.fill((0, 0, 0))
    appuye = False
    while True:
        if appuye:
            p = pygame.mouse.get_pos()
            pygame.draw.circle(fenetre, (255, 255, 255), p, 17)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                appuye = True
            if event.type == pygame.MOUSEBUTTONUP:
                appuye = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    fenetre.fill((0, 0, 0))
                if event.key == pygame.K_UP:
                    im = np.transpose(pygame.surfarray.array2d(fenetre)) / 65793
                    im = flou_gaussien(image_en_28x28(im, noir_sur_blanc=False), repetitions=2)
                    print(chr(np.argmax(N.Prediction(np.transpose(im).reshape(784))[-1][-1]) + 97))
                    #plt.imshow(im, cmap='gray')
                    #plt.show()

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

reconnaissance_en_direct()
