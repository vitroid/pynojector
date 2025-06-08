from pynojector import (
    flip,
    stack_from_imagestrip,
    imagestrip_from_mercator_ribbon,
    mercator_from_stereographic,
    translate,
    conversion,
)
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess
import tqdm


def projection(z: np.ndarray, shift: float) -> np.ndarray:

    # 関数名のつけかたに困惑している。
    # 一番左に、原画像の形式があり、一番右に、出力画像の形式がくるんだが、データの流れは逆向きなんだよね。

    z = flip(
        z=translate(
            z=stack_from_imagestrip(
                z=imagestrip_from_mercator_ribbon(
                    z=stack_from_imagestrip(
                        z=mercator_from_stereographic(z=translate(z=z, dx=-2, dy=-2)),
                        stack_height=9,
                    ),
                    theta=np.radians(45),
                ),
                stack_height=3,
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


filename = "/Users/matto/Dropbox/ArtsAndIllustrations/Stitch tmp2/DrYellow2/あっという間にゆきすぎる！ドクターイエローサイドビュー shorts.mp4.dir.16764.png"

img = Image.open(filename)
aspect_ratio = img.size[0] / img.size[1]
# print(aspect_ratio)

with tempfile.TemporaryDirectory() as temp_dir:
    duration = 3
    fps = 60
    frame_count = int(duration * fps)
    for i in tqdm.tqdm(range(frame_count)):
        r = (np.pi * 2) * aspect_ratio * (i / frame_count)
        projector = lambda z: projection(z=z, shift=r)
        # projector = lambda z: translate(z=z, dx=r, dy=0)
        dst_img = conversion(projector=projector, src_image=img, size=1440)
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
        dst_img.save(frame_path)

    save_movie(temp_dir, output=f"{filename}.swirl.mp4", fps=fps)
