import pygame
import numpy as np
from statistics import mean
from tqdm import tqdm # type: ignore

size = width, height = 1002, 667
win = pygame.display.set_mode(size)
pygame.display.set_caption("Mandelbrot Set")

def is_in_M(z, iterations=80): # Beau pour iterations dans [30, 80]
    p = np.sqrt((z.real-1/4)**2+z.imag**2)
    if z.real < p - 2*p**2 + 1/4:
        return 1
    if (z.real+1)**2 + z.imag**2 < 1/16:
        return 1
    u = z
    for i in range(iterations):
        u = u**2 + z
        if abs(u) > 2:
            return (i/iterations)**2
    return 1

start_color = np.array([0, 0, 0])
end_color = np.array([255, 255, 255])

def local_average(x, y, n=5): # lent à partir de n=3 mais plus n grand plus c'est beau
    L = []
    for i in range(n):
        for j in range(n):
            L.append(x+x_epsilon*i/n + 1j*(y+y_epsilon*j/n))
    return mean([is_in_M(e) for e in L])
    
def get_color(v):
    return (v*start_color[0] + (1-v)*end_color[0], v**2*start_color[1] + (1-v)**2*end_color[1], v**3*start_color[2] + (1-v)**3*end_color[2])

def resize_y(x_bounds, y_bounds):
    aim = (x_bounds[1] - x_bounds[0])*2/3
    middle = (y_bounds[1] + y_bounds[0]) / 2
    return (middle-aim/2, middle+aim/2)


drawn = False
running = True
pos = [[-2, 1], [1, -1]]
while running:
    if len(pos) == 2:
        x_bounds = tuple(sorted([pos[0][0], pos[1][0]]))
        y_bounds = tuple(sorted([pos[0][1], pos[1][1]]))
        y_bounds = resize_y(x_bounds, y_bounds)
        
        pos = []
        drawn = False
        
        win.fill((0, 0, 0))
        pygame.display.flip()

        x = np.linspace(x_bounds[0], x_bounds[1], width)
        y = np.linspace(y_bounds[0], y_bounds[1], height)

        x_epsilon = 0.5 * (x_bounds[1] - x_bounds[0]) / width
        y_epsilon = 0.5 * (y_bounds[1] - y_bounds[0]) / height
        
    if not drawn:
        for j in tqdm(range(height)):
            for i in range(width):
                v = local_average(x[i], y[j])
                pygame.draw.circle(win, get_color(v), (i, j), 1)
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
            
        