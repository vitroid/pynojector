import sys
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import scipy.special as sp
from tiffeditor import TiffEditor

# 長方形の画像の場合、左右を2とし、上下は周期境界とする。
# 中心を原点とする。
# 引数は複素数。
# 画像を拡大するのはz/2
# 中心を左に移動するのはz+0.5
# 中心を上に移動するのはz+0.5j
# 右無限遠を中央にもってきて放射状にするのはlog(z)
# Mercatorをequirectangularにするのは?


# 喩えば、放射状にしてから、中心を左に移動、その図をさらに拡大するのは、処理の順番としては、np.log(0.5+z/2)となる、はず。
# methodチェーンではなく、関数の関数として書くほうが直感的にわかりやすい。
@dataclass
class ProjectionChain:
    z: np.ndarray
    aspect_ratio: float


def mercator_from_equirectangular(z: np.ndarray) -> np.ndarray:
    # value rangeを-πからπにする。
    lon = z.real
    lat = z.imag
    return np.log(np.tan(lat / 2 + np.pi / 4)) * 1j + lon


def equirectangular_from_mercator(z: np.ndarray) -> np.ndarray:
    # value rangeを-πからπにする。
    lon = z.real
    lat = np.arctan(np.exp(z.imag)) * 2 - np.pi / 2
    return lon + 1j * lat


def mercator_from_stereographic(z: np.ndarray) -> np.ndarray:
    return np.log(z) * -1j


def stereographic_from_equirectangular(z: np.ndarray) -> np.ndarray:
    r = 2 * np.tan(np.pi / 4 - z.imag / 2)
    theta = z.real
    return r * np.exp(1j * theta)


def complex_ellipj(u, m):
    # x real part of z
    # y imaginary part
    # m parameter (note m=k2 where k is ellipse modulus)
    x, y = u.real, u.imag

    sn_x, cn_x, dn_x, ph_x = sp.ellipj(x, m)
    sn_y, cn_y, dn_y, ph_y = sp.ellipj(y, m)

    m_ = 1 - m
    sn_x_c, cn_x_x, dn_x_x, ph_x_c = sp.ellipj(x, m_)
    sn_y_c, cn_y_c, dn_y_c, ph_y_c = sp.ellipj(y, m_)

    common_den = cn_y_c**2 + m * sn_x**2 * sn_y_c**2

    complex_sn = sn_x * dn_y_c + 1j * sn_y_c * cn_y_c * cn_x * dn_x
    complex_sn /= common_den

    complex_cn = cn_x * cn_y_c - 1j * sn_x * dn_x * sn_y_c * dn_y_c
    complex_cn /= common_den

    complex_dn = dn_x * cn_y_c * dn_y_c - 1j * m * sn_x * cn_x * sn_y_c
    complex_dn /= common_den

    # TODO complex ph ?

    return complex_sn, complex_cn, complex_dn


def sd(u, m):
    sn, cn, dn = complex_ellipj(u, m)
    return sn / dn


def stereographic_from_peirce_quincuncial(z: np.ndarray) -> np.ndarray:
    # ソースは、https://ja.wikipedia.org/wiki/パース・クインカンシャル図法
    # 上下と左右でスケールが微妙に違うみたい。
    return sd(2**0.5 * z / np.pi, 0.5**0.5) / 2**0.5  # * np.pi


def translate(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return z - dx - 1j * dy


def scale(z: np.ndarray, magnify: float) -> np.ndarray:
    return z / magnify


def flip(z: np.ndarray) -> np.ndarray:
    return -z


def flip_x(z: np.ndarray) -> np.ndarray:
    return z.conj()


def flip_y(z: np.ndarray) -> np.ndarray:
    return -z.conj()


def equirectangular_rotation(z: np.ndarray, theta: float) -> np.ndarray:
    """
    z: equirectangular coordinate
    theta: rotation angle in degrees
    """
    lon, lat = z.real, z.imag
    # 緯度経度を球面座標に
    # latがpi/2を越えたら、pi/2にする。
    lat = np.clip(lat, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
    X = np.cos(lat) * np.cos(lon)
    Y = np.cos(lat) * np.sin(lon)
    Z = np.sin(lat)
    XYZ = np.array([X, Y, Z])
    # X軸回りの回転行列
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
            [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))],
        ]
    )
    XYZ_rot = R @ XYZ
    # 球面座標を緯度経度に
    lat_new = np.arcsin(XYZ_rot[2])
    lon_new = np.arctan2(XYZ_rot[1], XYZ_rot[0])
    return lon_new + 1j * lat_new


def stack_from_imagestrip(z: np.ndarray, stack_height: int) -> np.ndarray:
    r = z.real
    i = z.imag
    r *= 2 * stack_height + 1
    i *= 2 * stack_height + 1
    return r + 1j * i


def imagestrip_from_mercator_ribbon(z: np.ndarray, theta: float) -> np.ndarray:
    # 縦方向に無限に大きく、横幅は2piの画像の座標を、列車画像のピクセルに変換する。
    # 列車の傾きはthetaで与えられる。
    # 列車の高さに対する幅がaspect_ratioで与えられる。通常は1よりかなり大きい。

    # 1段の高さを決める。
    height1 = 2 * np.pi * np.tan(theta)
    # スロープの長さ
    slope_length = 2 * np.pi / np.cos(theta)
    # 画像の高さ
    ribbon_width = 2 * np.pi * np.sin(theta)
    # ribbon_length = ribbon_width * aspect_ratio
    # 以上、いずれもcanvas座標系での値。

    # 基本的な長さが得られたので、これをcanvas座標系から、画像の座標系に変換する。
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # canvas上の座標。中央を原点にする。
    x = z.real
    y = z.imag

    # スロープの段数
    story = np.floor(y / height1 - x / (2 * np.pi) + 0.5)
    y -= story * height1

    x1, y1 = R @ np.array([x, y])

    # x1_actual, y1は、imagestrip上の座標。
    x1_actual = x1 + story * slope_length

    # 最後に、新しいcanvas座標に変換
    x_image = x1_actual / ribbon_width * 2 * np.pi
    y_image = y1 / ribbon_width * 2 * np.pi

    # 新キャンバスでは、画像は縦が2pi、横はaspect_ratio*2pi

    return x_image + 1j * y_image


def conversion(
    projector: Callable[[np.ndarray, float], np.ndarray],
    src_image: np.ndarray | TiffEditor,
    size: int,
    dst_filename: str = None,
    two_by_one: bool = False,
) -> np.ndarray:
    h, w = src_image.shape[:2]
    src_array = src_image.reshape(-1, 3)

    if two_by_one:
        dst_array = np.zeros((size**2 // 2, 3), dtype=np.uint8)
    else:
        dst_array = np.zeros((size**2, 3), dtype=np.uint8)

    if two_by_one:
        X, Y = np.meshgrid(
            np.linspace(-np.pi, np.pi, size),
            np.linspace(-np.pi / 2, np.pi / 2, size // 2),
        )
    else:
        X, Y = np.meshgrid(
            np.linspace(-np.pi, np.pi, size), np.linspace(-np.pi, np.pi, size)
        )
    Z = (X + 1j * Y).reshape(-1)
    Z0 = projector(Z)
    Z1 = Z0 * h / (2 * np.pi)
    pix_x0 = np.round(Z1.real).astype(np.int32) + w // 2
    pix_y0 = np.round(Z1.imag).astype(np.int32) + h // 2
    pix_x0 %= w
    pix_y0 %= h
    pos = pix_x0 + pix_y0 * w
    dst_array[:] = src_array[pos, :]

    if two_by_one:
        dst_array = dst_array.reshape(size, size // 2, 3)
    else:
        dst_array = dst_array.reshape(size, size, 3)
    if dst_filename is not None:
        cv2.imwrite(dst_filename, dst_array)
    return dst_array
