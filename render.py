from typing import Tuple

import numpy as np
import pygame
from numba import njit, prange
from pygame.surface import Surface

scale = 350
maxiter = 150
cx = 0
cy = 0
SIZE = 1000
SIZE2 = SIZE / 2


@njit
def mandelbrot(c: np.complex64, maxiter: int):
    z = c
    for n in prange(maxiter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return 0


@njit(parallel=True)
def calculate(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    maxiter: int,
):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in prange(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], maxiter)
    return n3


def handle_input():
    global scale, cx, cy, maxiter
    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                quit()
            case pygame.KEYDOWN:
                match event.key:
                    case pygame.K_w:
                        scale *= 1.15
                        print(scale)
                        return True
                    case pygame.K_s:
                        scale /= 1.15
                        print(scale)
                        return True
                    case pygame.K_UP:
                        cy += 50 / scale
                        return True
                    case pygame.K_DOWN:
                        cy -= 50 / scale
                        return True
                    case pygame.K_LEFT:
                        cx += 50 / scale
                        return True
                    case pygame.K_RIGHT:
                        cx -= 50 / scale
                        return True
                    case pygame.K_EQUALS:
                        maxiter += 100
                        print(maxiter)
                        return False
                    case pygame.K_MINUS:
                        maxiter -= 100
                        print(maxiter)
                        return True
    return False


def update(screen: Surface, x: float, y: float, m_iter: int):
    xmin = x + cx - SIZE2 / scale
    xmax = x + cx + SIZE2 / scale
    ymin = y + cy - SIZE2 / scale
    ymax = y + cy + SIZE2 / scale

    ts = calculate(xmin, xmax, ymin, ymax, SIZE, SIZE, m_iter)

    for i in range(SIZE):
        for j in range(SIZE):
            t = ts[i, j]
            r, g, b = (int(t) * 3, int(t), int(t))
            screen.fill(
                (r % 255, g % 255, b % 255),
                (i, j, 3, 3),
            )
        changes = handle_input()
        if changes:
            pygame.display.update()
            return True

    pygame.display.update()
    return False


def start(screen: Surface):
    x = -0.8
    y = 0
    iter = 30
    # Game loop begins
    while True:
        step = (maxiter - iter) // 6
        if step > 0:
            screen.fill((255, 255, 255))

            iters = range(iter, maxiter, step)
            for m_iter in iters:
                changes = update(screen, x, y, m_iter)
                iter = m_iter
                if changes:
                    iter = 10
                    break
        else:
            changes = handle_input()
            if changes:
                iter = 10
