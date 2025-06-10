from pynojector import (
    flip,
    stack_from_imagestrip,
    imagestrip_from_mercator_ribbon,
    mercator_from_stereographic,
    translate,
    conversion,
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
    shift: float,
    bundle: int = 3,
    multiple: int = 9,
    tilt_angle: float = 45,
    centerx: float = 0,
    centery: float = 0,
) -> np.ndarray:

    # 関数名のつけかたに困惑している。
    # 一番左に、原画像の形式があり、一番右に、出力画像の形式がくるんだが、データの流れは逆向きなんだよね。

    z = flip(
        z=translate(
            z=stack_from_imagestrip(
                z=imagestrip_from_mercator_ribbon(
                    z=stack_from_imagestrip(
                        z=mercator_from_stereographic(
                            z=translate(
                                z=z, dx=centerx * 2 * np.pi, dy=centery * 2 * np.pi
                            )
                        ),
                        # z=mercator_from_stereographic(
                        #     z=stereographic_from_peirce_quincuncial(z=z)
                        # ),
                        stack_height=multiple,
                    ),
                    theta=np.radians(tilt_angle),
                ),
                stack_height=bundle,
            ),
            dx=-shift,
            dy=0,
        )
    )
    return z


def save_movie(
    temp_dir: str, output=None, encoder: str = "libx264", crf: int = 21, fps: int = 60
):
    cmd = [
        "ffmpeg",
        "-y",
        f"-framerate {fps}",
        f'-i "{temp_dir}/frame_%06d.jpg"',
        f"-c:v {encoder}",
        "-pix_fmt yuv420p",
        f"-crf {crf}" if crf else "",
        f'"{output}"',
    ]
    cmd = " ".join(cmd)
    print(cmd)
    subprocess.run(cmd, shell=True)


def movie_iter(
    img: np.ndarray,
    width: int = 1440,
    duration: float = 3.0,
    fps: int = 60,
    bundle: int = 3,
    multiple: int = 9,
    tilt_angle: float = 45,
    centerx: float = 0,
    centery: float = 0,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Ignoring: {kwargs}")
    aspect_ratio = img.shape[1] / img.shape[0]  # width / height
    frame_count = int(duration * fps)
    for i in tqdm.tqdm(range(frame_count)):
        shift = (np.pi * 2) * aspect_ratio * (i / frame_count)
        projector = lambda z: swirl_projection(
            z=z,
            shift=shift,
            bundle=bundle,
            multiple=multiple,
            tilt_angle=tilt_angle,
            centerx=centerx,
            centery=centery,
        )
        yield conversion(projector=projector, src_image=img, size=width)


def make_movie(
    img: np.ndarray,
    basename: str,
    width: int = 1440,
    duration: float = 3.0,
    fps: int = 60,
    bundle: int = 3,
    multiple: int = 9,
    tilt_angle: float = 45,
    centerx: float = 0,
    centery: float = 0,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Ignoring: {kwargs}")
    outfile = f"{basename}.swirl.mp4"
    aspect_ratio = img.shape[1] / img.shape[0]  # width / height
    frame_count = int(duration * fps)
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in tqdm.tqdm(range(frame_count)):
            shift = (np.pi * 2) * aspect_ratio * (i / frame_count)
            projector = lambda z: swirl_projection(
                z=z,
                shift=shift,
                bundle=bundle,
                multiple=multiple,
                tilt_angle=tilt_angle,
                centerx=centerx,
                centery=centery,
            )
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
            double_frame = conversion(
                projector=projector, src_image=img, size=width * 2
            )
            full_frame = cv2.resize(
                double_frame, (width, width), interpolation=cv2.INTER_CUBIC
            )
            cv2.imwrite(frame_path, full_frame)

        save_movie(temp_dir, output=outfile, fps=fps)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Input image file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of the movie (seconds)-- 1,100",
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second-- 1,120")
    parser.add_argument("--bundle", type=int, default=1, help="Number of bundles-- 0,5")
    parser.add_argument(
        "--multiple", type=int, default=6, help="Number of multiples-- 1,20"
    )
    parser.add_argument("--tilt_angle", type=float, default=45, help="Tilt angle--1,89")
    parser.add_argument(
        "--centerx", type=float, default=-0.33, help="Center of the swirl-- -1,1"
    )
    parser.add_argument(
        "--centery", type=float, default=-0.33, help="Center of the swirl-- -1,1"
    )
    parser.add_argument(
        "--width", type=int, default=1440, help="Width of the movie-- 100,10000"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    img = cv2.imread(args.filename)
    if img is None:
        raise ValueError("画像の読み込みに失敗しました")

    args_dict = vars(args)

    make_movie(
        img,
        basename=args.filename,
        **args_dict,
    )


if __name__ == "__main__":
    main()
