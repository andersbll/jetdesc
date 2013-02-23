import numpy as np
from numpy import pi, exp


class ScaleSpace:
    def __init__(self, img_shape, sigmas, dys, dxs):
        ''' Compute the scale-space of an image.
        Upon initialization, this class precomputes the Gaussian windows used
        to smooth images of a fixed shape to save the computations at later
        points.
        '''
        assert(len(sigmas) == len(dys) == len(dxs))
        h, w = img_shape
        g_y, g_x = np.mgrid[-.5 + .5 / h:.5:1. / h, -.5 + .5 / w:.5:1. / w]
        self.filters = []
        for sigma, dy, dx in zip(sigmas, dys, dxs):
            g = exp(- (g_x**2 + g_y**2) * (pi*2*sigma)**2 / 2.)
            g = np.fft.fftshift(g)
            if dy > 0 or dx > 0:
                dg_y = np.array((range(0, h/2)+range(-h/2, 0)), dtype=float,
                                ndmin=2) / h
                dg_x = np.array((range(0, w/2)+range(-w/2, 0)), dtype=float,
                                ndmin=2) / w
                dg = (dg_y.T ** dy) * (dg_x ** dx) * (1j * 2 * pi) ** (dy + dx)
                g = np.multiply(g, dg)
            self.filters.append(g)

    def compute_f(self, img_f):
        ''' Compute the scale space of an image in the fourier domain.'''
        return [np.multiply(img_f, f) for f in self.filters]

    def compute(self, img):
        ''' Compute the scale space of an image.'''
        img_f = np.fft.fft2(img)
        return [np.fft.ifft2(np.multiply(img_f, f)).real for f in self.filters]


def scale(img, sigma, dy=0, dx=0):
    '''Compute the scale-space of an image. sigma is the scale parameter. dx
    and dy specify the differentiation order along the x and y axis
    respectively.'''
    ss = ScaleSpace(img.shape, [sigma], [dy], [dx])
    return ss.compute(img)[0]
