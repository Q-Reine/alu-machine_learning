#!/usr/bin/env python3
"""
Defines a function that performs same convolution on a grayscale image
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    parameters:
        images [numpy.ndarray with shape (m, h, w)]:
            contains multiple grayscale images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
        kernel [numpy.ndarray with shape (kh, kw)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
    returns:
        numpy.ndarray contained convolved images
    """
    m, height, width = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    if (kh % 2) == 1:
        ph = (kh - 1) // 2
    else:
        ph = kh // 2
    if (kw % 2) == 1:
        pw = (kw - 1) // 2
    else:
        pw = kw // 2
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    convoluted = np.zeros((m, height, width))
    for h in range(height):
        for w in range(width):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
