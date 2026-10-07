import pygame
import numpy as np
from statistics import mean
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt
from time import time

size = width, height = 1002, 668
win = pygame.display.set_mode(size)
pygame.display.set_caption("Mandelbrot Set")

start_color = np.array([0, 0, 0])
end_color = np.array([255, 255, 255])
    
def get_color(v):
    c1 = v*start_color[0] + (1-v)*end_color[0]
    c2 = v**2*start_color[1] + (1-v)**2*end_color[1]
    c3 = v**3*start_color[2] + (1-v)**3*end_color[2]
    return (c1, c2, c3)

def compute_map(x_bounds, y_bounds, iterations=80, n=3): # Beau pour itérations dans [30, 80] et lent à partir de n=3 mais plus n grand plus c'est beau, prendre depassement = n/2
    x = np.linspace(x_bounds[0], x_bounds[1], width*n).reshape(1, width*n)
    x_tmp = np.ones((height*n)).reshape(height*n, 1)
    y = np.linspace(y_bounds[0], y_bounds[1], height*n).reshape(height*n, 1)
    y_tmp = 1j*np.ones((width*n)).reshape(1, width*n)
    
    map = np.matmul(x_tmp, x) + np.matmul(y, y_tmp)
    initial_map = np.copy(map)
    
    detailed_colors = np.zeros((height*n, width*n, 3), dtype=np.float64)
    for i in tqdm(range(iterations)):
        map = map**2+initial_map
        mask = (np.abs(map) > 2)
        map *= (1-mask)
        initial_map *= (1-mask)
        mask = mask*(np.sum(detailed_colors, axis=2) == 0)
        v = get_color((i/iterations)**2)
        detailed_colors[:,  :, 0] += mask * v[0]
        detailed_colors[:, :, 1] += mask * v[1]
        detailed_colors[:, :, 2] += mask * v[2]
        
    colors = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            colors[i, j] = np.mean(detailed_colors[n*i:n*i+n, n*j:n*j+n], axis=(0, 1))
    return colors

def resize_y(x_bounds, y_bounds):
    aim = (x_bounds[1] - x_bounds[0])*2/3
    middle = (y_bounds[1] + y_bounds[0]) / 2
    return (middle-aim/2, middle+aim/2)

drawn = False
running = True
pos = [[-2, 1], [1, -1]]
while running:
    if len(pos) == 2:
        # iterations = int(input("Itérations : "))
        # n = int(input("Zone de moyennage : "))
        x_bounds = tuple(sorted([pos[0][0], pos[1][0]]))
        y_bounds = tuple(sorted([pos[0][1], pos[1][1]]))
        y_bounds = resize_y(x_bounds, y_bounds)
        
        print(f"Bottom left corner : ({min(x_bounds)}, {min(y_bounds)}), Top right corner : ({max(x_bounds)}, {max(y_bounds)})")
        
        pos = []
        drawn = False
        
        win.fill((0, 0, 0))
        pygame.display.flip()
        
        colors = compute_map(x_bounds, y_bounds).astype(np.int32)
        
    if not drawn:
        for j in range(height):
            for i in range(width):
                pygame.draw.circle(win, colors[j, i], (i, j), 1)
            pygame.display.update((0, j, width, 2))
        drawn = True
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            p = list(pygame.mouse.get_pos())
            p[0] = x_bounds[0] + (x_bounds[1]-x_bounds[0])*p[0]/width
            p[1] = y_bounds[0] + (y_bounds[1]-y_bounds[0])*p[1]/height
            pos.append(p)
            
        
