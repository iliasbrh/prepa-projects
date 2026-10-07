import pygame, sys
import numpy as np
import scipy.integrate as integr
import cv2 as cv
pygame.init()
width, height = 600, 600
win = pygame.display.set_mode((width, height))

def background():
    win.fill((0, 0, 0))

win_offset_x, win_offset_y = width//2, height//2

def coefs_n(fonction, n): # encodé pour des fonctions sur [0, 1]
    tmp_r = lambda x : (fonction(x) * np.exp(-2j*n*np.pi*x)).real
    tmp_i = lambda x : (fonction(x) * np.exp(-2j*n*np.pi*x)).imag
    return round(integr.quad(tmp_r, 0, 1)[0], 4) + 1j * round(integr.quad(tmp_i, 0, 1)[0], 4)

def somme_fourier_N(L, t):
    N = len(L) // 2
    S = L[N]
    for n in range(1, N+1):
        terme = L[N+n] * np.exp(n*2j*np.pi*t)
        pygame.draw.line(win, (0, 255-(n/N)*220, 0), (S.real + win_offset_x, -S.imag + win_offset_y), ((S+terme).real + win_offset_x, -(S+terme).imag + win_offset_y))
        S += terme
        terme = L[N-n] * np.exp(-n * 2j * np.pi * t)
        pygame.draw.line(win, (0, 255-(n/N)*220, 0), (S.real + win_offset_x, -S.imag + win_offset_y),
                         ((S + terme).real + win_offset_x, -(S + terme).imag + win_offset_y))
        S += terme
    return S

def fonction_discrete_en_continue(liste, t): # encodé sur [0, 1]
    if t < 1:
        return liste[int(len(liste)*t)]
    elif t == 1:
        return liste[-1]

### A ajuster
nb_points_dessines = 1000
mode = "contours"
image = "img1.png"


float = 255/nb_points_dessines
points_dessines = []
L_points = []
appuye = False
t = 0
while True:
    if mode == "recording":
        if appuye:
            p = pygame.mouse.get_pos()
            pygame.draw.circle(win, (255, 255, 255), p, 1)
            points_dessines.append(p[0] - win_offset_x + (-p[1] + win_offset_y)*1j)
            print(points_dessines[-1])
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                appuye = True
            if event.type == pygame.MOUSEBUTTONUP:
                appuye = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    mode = "drawing"
                    f_construite = lambda t : fonction_discrete_en_continue(points_dessines, t)
                    N = 40
                    coefs = [coefs_n(f_construite, n) for n in range(-N, N + 1)]
                    res_fourier = lambda t : somme_fourier_N(coefs, t)

            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    elif mode == "contours":
        mode = "drawing"
        im = cv.imread(image)
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 50, 255, 0)
        contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0][1]
        abc = list((contours[:, :, 0] - 1j * contours[:, :, 1] - win_offset_x + 1j * win_offset_y)[:, 0])
        f_construite = lambda t : fonction_discrete_en_continue(abc, t)
        N = 50
        coefs = [coefs_n(f_construite, n) for n in range(-N, N + 1)]
        res_fourier = lambda t : somme_fourier_N(coefs, t)

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


    elif mode == "drawing":
        background()

        L_points.append(res_fourier(t))
        if len(L_points) > nb_points_dessines:
            L_points = L_points[1:]
        t += 0.001

        for i in range(len(L_points)):
            point = L_points[i]
            pygame.draw.circle(win, (255-(len(L_points)-i)*float, 255-(len(L_points)-i)*float, 255-(len(L_points)-i)*float), (point.real + win_offset_x, -point.imag + win_offset_y), 3)

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    N = int(input("N = "))
                    coefs = [coefs_n(f_construite, n) for n in range(-N, N+1)]
                    res_fourier = lambda t: somme_fourier_N(coefs, t)
                    L_points = []
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

















"""
def carre(t): # encodé sur [0, 1]
    if t < 0.25:
        return offset - t*400j*4
    elif t < 0.5:
        return offset - 400j + (t-0.25)*400*4
    elif t < 0.75:
        return -offset + (t-0.5)*400j*4
    elif t <= 1:
        return offset + (1-t)*400*4
"""
