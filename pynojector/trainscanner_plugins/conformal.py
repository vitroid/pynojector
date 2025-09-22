import argparse
import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
import tqdm

import pynojector as pj

__all__ = ["get_parser", "make_movie", "movie_iter"]


def conformal_projection(
    z: np.ndarray,
    tilt_angle: float = 45,
    finish: str = "pano",
    bundle: int = 1,
    multiple: int = 3,
    rotate: float = 0,
) -> np.ndarray:
    # 関数名のつけかたに困惑している。
    # 一番左に、原画像の形式があり、一番右に、出力画像の形式がくるんだが、データの流れは逆向きなんだよね。
    if finish == "pano":
        z = pj.mercator_from_equirectangular(z=pj.equirectangular_rotation(z, rotate))
    elif finish == "swirl":
        z = pj.mercator_from_equirectangular(
            z=pj.equirectangular_rotation(
                z=pj.equirectangular_from_mercator(
                    z=pj.mercator_from_stereographic(z=z)
                ),
                theta=rotate,
            )
        )
    elif finish == "square":
        z = pj.mercator_from_stereographic(
            z=pj.stereographic_from_peirce_quincuncial(z=z)
        )
    elif finish == "slant":
        pass
    else:
        raise ValueError(f"Invalid finish: {finish}")

    z = pj.stack_from_imagestrip(
        pj.imagestrip_from_mercator_ribbon(
            z=pj.stack_from_imagestrip(z=z, stack_height=multiple),
            theta=np.radians(tilt_angle),
        ),
        stack_height=bundle,
    )
    return z


def convert(
    img: np.ndarray,
    dst_filename: str = None,
    width: int = 1440,
    tilt_angle: float = 45,
    finish: str = "pano",
    bundle: int = 1,
    multiple: int = 3,
    rotate: float = 0,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.debug(f"Ignoring: {kwargs}")

    projector = lambda z: conformal_projection(
        z=z,
        tilt_angle=tilt_angle,
        finish=finish,
        bundle=bundle,
        multiple=multiple,
        rotate=rotate,
    )
    if finish == "pano":
        two_by_one = True
    else:
        two_by_one = False
    o = pj.conversion(
        projector=projector,
        src_image=img,
        dst_filename=dst_filename,
        size=width,
        two_by_one=two_by_one,
    )
    return o


def get_parser():
    parser = argparse.ArgumentParser(description="Conformal transformations")
    parser.add_argument("filename", type=str, help="Input image file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--finish",
        type=str,
        default="pano",
        help="Type of the image -- pano|swirl|slant",
    )
    parser.add_argument("--bundle", type=int, default=1, help="Bundle -- 0:5")
    parser.add_argument(
        "--tilt_angle", type=float, default=15, help="Tilt angle -- 0.5:89"
    )
    parser.add_argument("--multiple", type=int, default=3, help="Multiple -- 0:10")
    parser.add_argument("--rotate", type=float, default=0, help="Rotate -- 0:90")
    parser.add_argument(
        "--width", type=int, default=4000, help="Width of the panorama -- 400:20000"
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
    cv2.imwrite(args.filename + ".conformal.jpg", output)


if __name__ == "__main__":
    main()
