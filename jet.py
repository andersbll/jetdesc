import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.linalg import svd
from scipy.misc import factorial

from scalespace import ScaleSpace
from util import extract_keypoint


class JetDescriptor:
    def __init__(self, k=4, sigma=5.3, rings=1, ring_samplings=4,
                 normalization='l2', whitening=True, patch_size=64,
                 keypoint_scale=3):

        self.whitening = whitening
        self.keypoint_scale = keypoint_scale
        self.normalization = normalization
        self.jet_dim = self.jet_dimensionality(k)-1
        self.desc_dim = (rings * ring_samplings + 1) * self.jet_dim
        self.patch_shape = (patch_size, patch_size)

        # Generate Fourier filters
        dys = []
        dxs = []
        self.orders = []
        for order in range(1, k+1):
            for i in range(order+1):
                dys.append(i)
                dxs.append(order-i)
                self.orders.append(order)
        self.sigmas = [sigma] * self.jet_dim
        self.scalespace = ScaleSpace(self.patch_shape, self.sigmas, dys, dxs)

        # Calculate sampling points
        center = float(patch_size)/2
        self.x_coords = [int(round(center))]
        self.y_coords = [int(round(center))]
        for r in range(rings):
            for i in range(ring_samplings):
                theta = 2 * pi * (i + 0.5*(r % 2)) / ring_samplings
                dist = sigma*2*(r+1)
                self.y_coords.append(int(round(center + dist * sin(theta))))
                self.x_coords.append(int(round(center + dist * cos(theta))))

        # Jet whitening
        if whitening:
            covar = np.zeros((self.jet_dim, self.jet_dim))
            for i in range(self.jet_dim):
                for j in range(self.jet_dim):
                    m = dys[j] + dys[i]
                    n = dxs[j] + dxs[i]
                    if not (n & 1 or m & 1):
                        order = float(m+n)
                        covar[i, j] = (-1.0)**(order / 2 + (dys[j]+dxs[j])) \
                                      * factorial(n) * factorial(m) \
                                      / (2 * pi * 2**order * order *
                                         factorial(n/2) * factorial(m/2))
            V, D, _ = svd(covar)
            self.whitener = np.dot(V, np.diag(D**(-.5)))

    def jet_dimensionality(self, k):
        return int(factorial(2+k)/(2*factorial(k)))

    def compute(self, img, keypoints):
        descs = np.empty((len(keypoints), len(self.y_coords), self.jet_dim))
        patch_jet = np.empty((self.jet_dim,) + self.patch_shape)

        for k, keypoint in enumerate(list(keypoints)):
            keypoint[:2] = keypoint[:2]-1
            patch = extract_keypoint(img, keypoint, self.patch_shape,
                                     self.keypoint_scale)
            # Compute image jets
            derivs = self.scalespace.compute(patch)
            for i in range(len(derivs)):
                patch_jet[i, :, :] = self.sigmas[i]**self.orders[i]*derivs[i]

            # Extract jet samples
            X = patch_jet[:, self.x_coords, self.y_coords]
            descs[k, :, :] = X.T

        # Whitening
        descs = np.dot(descs, self.whitener)
        descs = np.reshape(descs, (len(keypoints), self.desc_dim))

        # Normalize descriptors.
        if self.normalization != 'off':
            descs += 1e-10
            if self.normalization == 'l1':
                descs /= np.sum(descs, axis=1)[:, np.newaxis]
            elif self.normalization == 'l2':
                descs /= sqrt(np.sum(descs ** 2, axis=1))[:, np.newaxis]

        return descs
