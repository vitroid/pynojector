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
from PIL import Image
import tempfile
import os
import subprocess
import tqdm
import argparse


def swirl_projection(
    z: np.ndarray,
    shift: float,
    bundle: int = 3,
    multiple: int = 9,
    tilt_angle: float = 45,
    center: tuple[float, float] = (0, 0),
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
                                z=z, dx=center[0] * 2 * np.pi, dy=center[1] * 2 * np.pi
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


def swirl_movie(
    img: Image.Image,
    temp_dir: str,
    outfile: str,
    duration: float = 3.0,
    fps: int = 60,
    bundle: int = 3,
    multiple: int = 9,
    tilt_angle: float = 45,
    center: tuple[float, float] = (0, 0),
):
    aspect_ratio = img.size[0] / img.size[1]
    frame_count = int(duration * fps)
    for i in tqdm.tqdm(range(frame_count)):
        shift = (np.pi * 2) * aspect_ratio * (i / frame_count)
        projector = lambda z: swirl_projection(
            z=z,
            shift=shift,
            bundle=bundle,
            multiple=multiple,
            tilt_angle=tilt_angle,
            center=center,
        )
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
        conversion(projector=projector, src_image=img, size=1440).save(frame_path)

    save_movie(temp_dir, output=outfile, fps=fps)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Input image file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Duration of the movie (seconds)"
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--bundle", type=int, default=3, help="Number of bundles")
    parser.add_argument("--multiple", type=int, default=9, help="Number of multiples")
    parser.add_argument("--tilt_angle", type=float, default=45, help="Tilt angle")
    parser.add_argument(
        "--centerx", type=float, default=(0, 0), help="Center of the swirl"
    )
    parser.add_argument(
        "--centery", type=float, default=(0, 0), help="Center of the swirl"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    filename = args.filename
    debug = args.debug
    duration = args.duration
    fps = args.fps
    bundle = args.bundle
    multiple = args.multiple
    tilt_angle = args.tilt_angle
    centerx = args.centerx
    centery = args.centery
    center = (centerx, centery)

    img = Image.open(filename)
    outfile = f"{filename}.swirl.mp4"
    # print(aspect_ratio)
    if debug:
        workarea = "."
    else:
        workarea = None

    with tempfile.TemporaryDirectory(dir=workarea) as temp_dir:
        swirl_movie(
            img,
            temp_dir,
            outfile,
            duration=duration,
            fps=fps,
            bundle=bundle,
            multiple=multiple,
            tilt_angle=tilt_angle,
            center=center,
        )


if __name__ == "__main__":
    main()
