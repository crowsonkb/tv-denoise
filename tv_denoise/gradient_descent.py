"""Total variation denoising."""

from dataclasses import dataclass

import numpy as np


RGB_TO_YUV = np.float32([
    [0.2126, 0.7152, 0.0722],
    [-0.09991, -0.33609, 0.436],
    [0.615, -0.55861, -0.05639],
])

YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV)


def tv_norm(image, eps=1e-8):
    """Computes the isotropic total variation norm and its gradient. Modified from
    https://github.com/jcjohnson/cnn-vis."""
    x_diff = image[:-1, :-1, ...] - image[:-1, 1:, ...]
    y_diff = image[:-1, :-1, ...] - image[1:, :-1, ...]
    grad_mag = np.sqrt(x_diff**2 + y_diff**2 + eps)
    loss = np.sum(grad_mag)
    dx_diff = x_diff / grad_mag
    dy_diff = y_diff / grad_mag
    grad = np.zeros_like(image)
    grad[:-1, :-1, ...] = dx_diff + dy_diff
    grad[:-1, 1:, ...] -= dx_diff
    grad[1:, :-1, ...] -= dy_diff
    return loss, grad


def l2_norm(image, orig_image):
    """Computes 1/2 the square of the L2-norm of the difference between the image and
    the original image and its gradient."""
    grad = image - orig_image
    loss = np.sum(grad**2) / 2
    return loss, grad


def eval_loss_and_grad(image, orig_image, strength_luma, strength_chroma):
    """Computes the loss function for TV denoising and its gradient."""
    tv_loss_y, tv_grad_y = tv_norm(image[:, :, 0])
    tv_loss_uv, tv_grad_uv = tv_norm(image[:, :, 1:])
    tv_grad = np.zeros_like(image)
    tv_grad[..., 0] = tv_grad_y * strength_luma
    tv_grad[..., 1:] = tv_grad_uv * strength_chroma
    l2_loss, l2_grad = l2_norm(image, orig_image)
    loss = tv_loss_y * strength_luma + tv_loss_uv * strength_chroma + l2_loss
    grad = tv_grad + l2_grad
    return loss, grad


@dataclass
class GradientDescentDenoiseStatus:
    """A status object supplied to the callback specified in tv_denoise_gradient_descent()."""
    i: int
    loss: float


# pylint: disable=too-many-arguments, too-many-locals
def tv_denoise_gradient_descent(image,
                                strength_luma,
                                strength_chroma,
                                callback=None,
                                step_size=1e-2,
                                tol=3.2e-3):
    """Total variation image denoising with gradient descent."""
    image = image @ RGB_TO_YUV.T
    orig_image = image.copy()
    momentum = np.zeros_like(image)
    momentum_beta = 0.9
    loss_smoothed = 0
    loss_smoothing_beta = 0.9
    i = 0
    while True:
        i += 1

        loss, grad = eval_loss_and_grad(image, orig_image, strength_luma, strength_chroma)

        if callback is not None:
            callback(GradientDescentDenoiseStatus(i, loss))

        # Stop iterating if the loss has not been decreasing recently
        loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1 - loss_smoothing_beta)
        loss_smoothed_debiased = loss_smoothed / (1 - loss_smoothing_beta**i)
        if i > 1 and loss_smoothed_debiased / loss < tol + 1:
            break

        # Calculate the step size per channel
        step_size_luma = step_size / (strength_luma + 1)
        step_size_chroma = step_size / (strength_chroma + 1)
        step_size_arr = np.float32([[[step_size_luma, step_size_chroma, step_size_chroma]]])

        # Gradient descent step
        momentum *= momentum_beta
        momentum += grad * (1 - momentum_beta)
        image -= step_size_arr / (1 - momentum_beta**i) * momentum

    return image @ YUV_TO_RGB.T
