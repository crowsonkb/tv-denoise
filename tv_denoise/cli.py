#!/usr/bin/env python3

"""Total variation denoising for images."""

import argparse

from PIL import Image

from tv_denoise import to_float32, to_uint8, tv_denoise_chambolle, tv_denoise_gradient_descent


def main():
    """The main function."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='the input image')
    ap.add_argument('output', help='the output image')
    ap.add_argument('strength', type=float, nargs='?', default=0.1,
                    help='the luma denoising strength')
    ap.add_argument('strength_chroma', type=float, nargs='?',
                    help='the chroma denoising strength (gradient descent method only, twice '
                         'luma strength if not specified)')
    ap.add_argument('--method', default='gradient', choices=['gradient', 'chambolle'],
                    help='the denoising method to use')
    args = ap.parse_args()

    if args.strength_chroma is None:
        args.strength_chroma = args.strength * 2

    def grad_callback(status):
        print(f'step: {status.i}, loss: {status.loss:g}')

    def chambolle_callback(status):
        print(f'step: {status.i}, max diff: {status.diff:g}')

    image = to_float32(Image.open(args.input).convert('RGB'))

    if args.method == 'gradient':
        out_arr = tv_denoise_gradient_descent(image,
                                              args.strength,
                                              args.strength_chroma,
                                              callback=grad_callback)
    elif args.method == 'chambolle':
        out_arr = tv_denoise_chambolle(image, args.strength, callback=chambolle_callback)
    else:
        raise ValueError('Invalid method')

    out = Image.fromarray(to_uint8(out_arr))
    out.save(args.output)


if __name__ == '__main__':
    main()
