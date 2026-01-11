#!/usr/bin/env python3
"""class NST"""
import tensorflow as tf
import numpy as np


class NST:
    """Class NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        style_image - the image used as a style reference
        content_image - the image used as a content reference
        alpha - the weight for content cost
        beta - the weight for style cost
        """

        if (not isinstance(style_image, np.ndarray)
                or len(style_image.shape) != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if (not isinstance(content_image, np.ndarray)
                or len(content_image.shape) != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError('alpha must be a non-negative number')

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError('beta must be a non-negative number')

        tf.config.run_functions_eagerly(True)

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        """

        if (not isinstance(image, np.ndarray)
                or len(image.shape) != 3
                or image.shape[2] != 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )
        max_dim = 512
        h, w, _ = image.shape
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize(
            image, (new_h, new_w), method='bicubic'
        )
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return image
