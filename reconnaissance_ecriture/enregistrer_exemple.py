import pygame, sys
import numpy as np
import matplotlib.image
from pretraitement_images import flou_gaussien
from image28x28 import image_en_28x28
fenetre = pygame.display.set_mode((560, 560))
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
                im = flou_gaussien(image_en_28x28(im, noir_sur_blanc=False), repetitions=1)
                nom = "Lettres\Images_Lettres\image_" + input('Nom du fichier : ') + '.png'
                matplotlib.image.imsave(nom, im)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()