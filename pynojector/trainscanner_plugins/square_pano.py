import argparse
import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
import tqdm

from pynojector import (conversion, imagestrip_from_mercator_ribbon,
                        mercator_from_stereographic,
                        stereographic_from_peirce_quincuncial)

__all__ = ["get_parser", "make_movie", "movie_iter"]


def swirl_projection(
    z: np.ndarray,
    tilt_angle: float = 45,
) -> np.ndarray:
    # 関数名のつけかたに困惑している。
    # 一番左に、原画像の形式があり、一番右に、出力画像の形式がくるんだが、データの流れは逆向きなんだよね。

    z = imagestrip_from_mercator_ribbon(
        z=mercator_from_stereographic(z=stereographic_from_peirce_quincuncial(z=z)),
        theta=np.radians(tilt_angle),
    )
    return z


def convert(
    img: np.ndarray,
    width: int = 1440,
    tilt_angle: float = 45,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.debug(f"Ignoring: {kwargs}")

    projector = lambda z: swirl_projection(
        z=z,
        tilt_angle=tilt_angle,
    )
    o = conversion(projector=projector, src_image=img, size=width)
    return o


def get_parser():
    parser = argparse.ArgumentParser(description="Make a square panorama image (1:1)")
    parser.add_argument("filename", type=str, help="Input image file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--tilt_angle", type=float, default=15, help="Tilt angle--0.5,89"
    )
    parser.add_argument(
        "--width", type=int, default=4000, help="Width of the panorama -- 400,20000"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    img = cv2.imread(args.filename)
    if img is None:
        raise ValueError("画像の読み込みに失敗しました")

    args_dict = vars(args)

    output = convert(
        img,
        basename=args.filename,
        **args_dict,
    )
    # always 2:1 aspect ratio
    cv2.imwrite(args.filename + ".square.jpg", output)


if __name__ == "__main__":
    main()
