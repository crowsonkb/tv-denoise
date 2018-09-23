"""Utility functions."""

import numpy as np


def to_float32(image):
    """Converts a uint8 image, in numpy or Pillow format, to float32."""
    return np.float32(image) / 255


def to_uint8(image):
    """Converts a float32 image to a numpy uint8 image."""
    return np.uint8(np.round(np.clip(image, 0, 1) * 255))
