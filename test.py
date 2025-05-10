from PIL import Image
import sys
import numpy as np
from typing import Callable
import scipy.special as sp

img = Image.open(sys.argv[1])

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
    return sd(2**0.5 * z, 0.5**0.5) / 2**0.5


def translate(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return z - dx - 1j * dy


def scale(z: np.ndarray, magnify: float) -> np.ndarray:
    return z / magnify


def equirectangular_rotation(z: np.ndarray, theta: float) -> np.ndarray:
    """
    z: equirectangular coordinate
    theta: rotation angle in radians
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
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    XYZ_rot = R @ XYZ
    # 球面座標を緯度経度に
    lat_new = np.arcsin(XYZ_rot[2])
    lon_new = np.arctan2(XYZ_rot[1], XYZ_rot[0])
    print(lon[:10], lon_new[:10])
    print(lat[:10], lat_new[:10])
    return lon_new + 1j * lat_new


def projection(z: np.ndarray) -> np.ndarray:
    # return mercator_from_equirectangular(equirectangular_from_mercator(z))
    # return equirectangular_from_mercator(z)
    # return z
    # return stereographic_from_equirectangular(equirectangular_from_stereographic(z))
    # return np.log(0.5 + z / 2)
    # return mercator_from_stereographic(
    #     stereographic_from_peirce_quincuncial(z)
    # )
    return mercator_from_equirectangular(equirectangular_rotation(z, np.radians(30)))
    # return mercator_from_equirectangular(z)


def conversion(
    projector: Callable[[np.ndarray], np.ndarray], src_image: Image.Image, size: int
) -> Image.Image:
    w, h = src_image.size
    src_array = np.array(src_image).reshape(-1, 3)

    dst_array = np.zeros((size**2, 3), dtype=np.uint8)

    X, Y = np.meshgrid(
        np.linspace(-np.pi, np.pi, size), np.linspace(-np.pi, np.pi, size)
    )
    Z = (X + 1j * Y).reshape(-1)
    Z0 = projector(Z)
    Z1 = Z0 * w / (2 * np.pi)
    pix_x0 = np.round(Z1.real).astype(np.int32) + w // 2
    pix_y0 = np.round(Z1.imag).astype(np.int32) + h // 2
    pix_x0 %= w
    pix_y0 %= h
    pos = pix_x0 + pix_y0 * w
    dst_array[:] = src_array[pos, :]

    return Image.fromarray(dst_array.reshape(size, size, 3))


result_img = conversion(projection, img, size=2000)
result_img.save("result.png")
