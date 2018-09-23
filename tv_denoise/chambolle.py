"""Implements Chambolle's projection algorithm for total variation image denoising. See
https://www.ipol.im/pub/art/2013/61/article.pdf."""

from dataclasses import dataclass

import numpy as np


def grad(arr):
    """Computes the discrete gradient of an image."""
    out = np.zeros((2,) + arr.shape, arr.dtype)
    out[0, :-1, :, ...] = arr[1:, :, ...] - arr[:-1, :, ...]
    out[1, :, :-1, ...] = arr[:, 1:, ...] - arr[:, :-1, ...]
    return out


def div(arr):
    """Computes the discrete divergence of a vector array."""
    out = np.zeros_like(arr)
    out[0, 0, :, ...] = arr[0, 0, :, ...]
    out[0, -1, :, ...] = -arr[0, -2, :, ...]
    out[0, 1:-1, :, ...] = arr[0, 1:-1, :, ...] - arr[0, :-2, :, ...]
    out[1, :, 0, ...] = arr[1, :, 0, ...]
    out[1, :, -1, ...] = -arr[1, :, -2, ...]
    out[1, :, 1:-1, ...] = arr[1, :, 1:-1, ...] - arr[1, :, :-2, ...]
    return np.sum(out, axis=0)


def magnitude(arr, axis=0, keepdims=False):
    """Computes the element-wise magnitude of a vector array."""
    return np.sqrt(np.sum(arr**2, axis=axis, keepdims=keepdims))


@dataclass
class ChambolleDenoiseStatus:
    """A status object supplied to the callback specified in tv_denoise_chambolle()."""
    i: int
    diff: float


def tv_denoise_chambolle(image, strength, step_size=0.25, tol=3.2e-3, callback=None):
    """Total variation image denoising with Chambolle's projection algorithm."""
    image = np.atleast_3d(image)
    p = np.zeros((2,) + image.shape, image.dtype)
    image_over_strength = image / strength
    diff = np.inf
    i = 0
    while diff > tol:
        i += 1
        grad_div_p_i = grad(div(p) - image_over_strength)
        mag_gdpi = magnitude(grad_div_p_i, axis=(0, -1), keepdims=True)
        new_p = (p + step_size * grad_div_p_i) / (1 + step_size * mag_gdpi)
        diff = np.max(magnitude(new_p - p))
        if callback is not None:
            callback(ChambolleDenoiseStatus(i, float(diff)))
        p[:] = new_p

    return np.squeeze(image - strength * div(p))
