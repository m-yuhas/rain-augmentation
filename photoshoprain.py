"""Module for bulk generation of rainy images following the algorithm in
https://www.photoshopessentials.com/photo-effects/photoshop-weather-effects-rain/,
which appeared in:

X. Fu, Q. Qi, Z. -J. Zha, X. Ding, F. Wu, and J. Paisley, ``Successive graph
convolutional network for image de-raining,'' International Journal of Computer
Vision, vol. 129, pp. 1691--1711, May 2021, doi: 10.1007/s11263-020-01428-6.

H. Yin, F. Zheng, H. -F. Duan, D. Savic, and Z. Kapelan, ``Estimating Rainfall
Intensity Using an Image-Based Deep Learning Model,'' Engineering, vol. 21,
pp. 162--174, Feb. 2023, doi: 10.1016/j.eng.2021.11.021.


L. Wang, H. Qin, X. Zhou, X. Lu and F. Zhang, ``R-YOLO: A Robust Object
Detector in Adverse Weather,'' IEEE Transactions on Instrumentation and
Measurement, vol. 72, pp. 1--11, Dec. 2022, doi: 10.1109/TIM.2022.3229717.
"""

from typing import Tuple
import argparse
import os

import cv2
import numpy


def get_rain_mask(size: Tuple[int, int],
                  amount: float = 25,
                  angle: float = 65,
                  drop_length: int = 75,
                  drop_size: int = 4,
                  black_point: int = 0,
                  white_point: int = 255) -> numpy.ndarray:
    mask = numpy.random.normal(loc=amount, scale=255 / 3, size=size)
    print(mask)
    mask = cv2.resize(mask, (size[1] * drop_size + 2 * drop_length, size[0] * drop_size + 2 * drop_length))
    mask = mask[:size[0] + 2 * drop_length, :size[1] + 2 * drop_length]
    kernel = numpy.zeros((drop_length, drop_length))
    kernel[drop_length // 2, :] = numpy.ones(drop_length, dtype=numpy.float32)
    rotation_matrix = cv2.getRotationMatrix2D((drop_length / 2, drop_length /2), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, kernel.shape[::-1])
    kernel = kernel * (1.0 / numpy.sum(kernel))
    mask = cv2.filter2D(mask, -1, kernel)
    mask = mask[drop_length:size[0]+drop_length, drop_length:size[1]+drop_length]
    mask = 255 * (mask - black_point) / (white_point - black_point)
    mask = numpy.clip(mask, 0, 255)
    mask = numpy.repeat(mask[:, :, numpy.newaxis], 3, axis=2)
    return mask

def apply_mask(img, mask):
    img = img.astype(numpy.float32) / 255
    mask = mask / 255
    img = 1 - (1 - img) * (1 - mask)
    return (img * 255).astype(numpy.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Apply photoshop rain to photo.')
    parser.add_argument(
        '--dataset',
        help='Path to images to which to apply rain.',
    )
    parser.add_argument(
        '--amount',
        type=float,
        help='Rain intensity'
    )
    parser.add_argument(
        '--angle',
        type=float,
        help='Rain angle',
    )
    parser.add_argument(
        '--drop_length',
        type=int,
        help='Length of rain drops'
    )
    parser.add_argument(
        '--drop_size',
        type=int,
        help='Size of rain drops'
    )
    parser.add_argument(
        '--black_point',
        type=int,
        help='Increase this to increase rain constrast'
    )
    parser.add_argument(
        '--white_point',
        type=int,
        help='Decrease this to increase rain contrast'
    )
    args = parser.parse_args()
    os.mkdir(args.dataset + f'_rain{args.amount}')
    for f in os.listdir(args.dataset):
        img = cv2.imread(os.path.join(args.dataset, f))
        mask = get_rain_mask(
            size=img.shape[0:2],
            amount=args.amount,
            angle=args.angle,
            drop_length=args.drop_length,
            drop_size=args.drop_size,
            black_point=args.black_point,
            white_point=args.white_point,
        )
        img = apply_mask(img, mask)
        cv2.imwrite(os.path.join(args.dataset + f'_rain{args.amount}', f), img)
