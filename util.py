import numpy as np
from numpy import sqrt, pi, arctan2
from numpy.linalg import eig
from scipy.ndimage.interpolation import affine_transform
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def read_keypoints(path):
    with open(path, 'r') as f:
        f.readline()
        num_keypoints = int(float(f.readline()))
        data = np.loadtxt(f)
    assert(num_keypoints == data.shape[0])
    return data


def write_keypoints(path, keypoints, descs=None):
    if descs is not None:
        data = np.concatenate((keypoints, descs), axis=1)
        desc_dim = descs.shape[1]
    else:
        data = keypoints
        desc_dim = 1
    with open(path, 'w') as f:
        f.write(str(desc_dim)+'\n')
        f.write(str(data.shape[0])+'\n')
        np.savetxt(f, data, fmt='%.12f')


def draw_keypoint(keypoint, scale):
    x = keypoint[0]
    y = keypoint[1]
    a = keypoint[2]
    b = keypoint[3]
    c = keypoint[4]
    A = np.array([[a, b], [b, c]])
    w, v = eig(A)
    width = 1/sqrt(w[0])*2 * scale
    height = 1/sqrt(w[1])*2 * scale
    angle = arctan2(v[1, 0], v[1, 1])/pi*180
    ellipse = Ellipse((x, y), width=width, height=height, angle=angle,
                      facecolor='none', edgecolor="black", linewidth=3)
    plt.gca().add_patch(ellipse)
    ellipse = Ellipse((x, y), width=width, height=height, angle=angle,
                      facecolor='none', edgecolor="yellow", linewidth=1)
    plt.gca().add_patch(ellipse)


def extract_keypoint(img, keypoint, patch_shape, scale):
    x = keypoint[0]
    y = keypoint[1]
    a = keypoint[2]
    b = keypoint[3]
    c = keypoint[4]
    A = np.array([[c, b], [b, a]])
    u, s, v = np.linalg.svd(A)
    s = 1/np.sqrt(s)
    A = np.dot(u, np.dot(np.diag(s), v))
    A = A * scale * 2/patch_shape[0]

    offset = np.array([y, x])
    offset -= np.dot(A, np.array([patch_shape[0], patch_shape[1]]))/2
    patch = affine_transform(img, A, offset=offset, output_shape=patch_shape,
                             order=1, prefilter=False)
    return patch
