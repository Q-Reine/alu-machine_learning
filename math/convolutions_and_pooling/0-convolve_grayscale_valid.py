#!/usr/bin/env python3
"""
Defines a function that performs valid convolution
on a grayscale image
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

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
    convoluted = np.zeros((m, height - kh + 1, width - kw + 1))
    for h in range(height - kh + 1):
        for w in range(width - kw + 1):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
