from pynojector import (
    imagestrip_from_mercator_ribbon,
    conversion,
    mercator_from_equirectangular,
    mercator_from_stereographic,
    stereographic_from_peirce_quincuncial,
)
import numpy as np
import cv2
import tempfile
import os
import subprocess
import tqdm
import argparse
import logging

__all__ = ["get_parser", "make_movie", "movie_iter"]


def swirl_projection(
    z: np.ndarray,
    tilt_angle: float = 45,
    finish: str = "pano",
) -> np.ndarray:

    # 関数名のつけかたに困惑している。
    # 一番左に、原画像の形式があり、一番右に、出力画像の形式がくるんだが、データの流れは逆向きなんだよね。
    if finish == "pano":
        z = mercator_from_equirectangular(z=z)
    elif finish == "swirl":
        z = mercator_from_stereographic(z=z)
    elif finish == "square":
        z = mercator_from_stereographic(z=stereographic_from_peirce_quincuncial(z=z))
    else:
        raise ValueError(f"Invalid finish: {finish}")

    z = imagestrip_from_mercator_ribbon(
        z=z,
        theta=np.radians(tilt_angle),
    )
    return z


def convert(
    img: np.ndarray,
    width: int = 1440,
    tilt_angle: float = 45,
    finish: str = "pano",
    **kwargs,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.debug(f"Ignoring: {kwargs}")

    projector = lambda z: swirl_projection(
        z=z,
        tilt_angle=tilt_angle,
        finish=finish,
    )
    o = conversion(projector=projector, src_image=img, size=width)
    if finish == "pano":
        o = o[width // 4 : width * 3 // 4, :]
    return o


def get_parser():
    parser = argparse.ArgumentParser(description="Make a swirl image")
    parser.add_argument("filename", type=str, help="Input image file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--tilt_angle", type=float, default=15, help="Tilt angle -- 0.5:89"
    )
    parser.add_argument(
        "--width", type=int, default=4000, help="Width of the panorama -- 400:20000"
    )
    parser.add_argument(
        "--finish",
        type=str,
        default="pano",
        help="Type of the image -- pano|swirl|square",
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
    cv2.imwrite(args.filename + ".swirl.jpg", output)


if __name__ == "__main__":
    main()
