#!/usr/bin/env python3

"""Total variation denoising for images."""

import argparse

from PIL import Image

from .tv_denoise import to_float32, to_uint8, tv_denoise


def main():
    """The main function."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='the input image')
    ap.add_argument('output', help='the output image')
    ap.add_argument('strength_luma', type=float, nargs='?', default=0.1,
                    help='the luma denoising strength')
    ap.add_argument('strength_chroma', type=float, nargs='?',
                    help='the chroma denoising strength (twice luma strength if not specified)')
    args = ap.parse_args()

    if args.strength_chroma is None:
        args.strength_chroma = args.strength_luma * 2

    def callback(status):
        print(f'step: {status.i}, loss: {status.loss:g}')

    image = to_float32(Image.open(args.input).convert('RGB'))
    out_arr = tv_denoise(image, args.strength_luma, args.strength_chroma, callback=callback)
    out = Image.fromarray(to_uint8(out_arr))
    out.save(args.output)


if __name__ == '__main__':
    main()
