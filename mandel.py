import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit, prange


@njit
def mandelbrot(c, maxiter):
    z = c
    for n in prange(maxiter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return 0


@njit
def mandelbrot_set_base(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1, r2, [mandelbrot(complex(r, i), maxiter) for r in r1 for i in r2])


@njit(parallel=True)
def mandelbrot_set_np(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in prange(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], maxiter)
    return (r1, r2, n3)


mandelbrot_set = mandelbrot_set_np

# mandelbrot_set(-2.0, 0.5, -1.25, 1.25, 1000, 1000, 80)
mandelbrot_set(-0.74877, -0.74872, 0.06505, 0.06510, 1000, 1000, 2048)
