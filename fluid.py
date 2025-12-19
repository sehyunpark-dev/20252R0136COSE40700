import math

import warp as wp
import warp.render

grid_width = wp.constant(256)
grid_height = wp.constant(128)

@wp.func
def lookup_float(f: wp.array2d(dtype=float), x: int, y: int):
    x = wp.clamp(x, 0, grid_width - 1)
    y = wp.clamp(y, 0, grid_height - 1)

    return f[x, y]


@wp.func
def sample_float(f: wp.array2d(dtype=float), x: float, y: float):
    lx = int(wp.floor(x))
    ly = int(wp.floor(y))

    tx = x - float(lx)
    ty = y - float(ly)

    s0 = wp.lerp(lookup_float(f, lx, ly), lookup_float(f, lx + 1, ly), tx)
    s1 = wp.lerp(lookup_float(f, lx, ly + 1), lookup_float(f, lx + 1, ly + 1), tx)

    s = wp.lerp(s0, s1, ty)
    return s


@wp.func
def lookup_vel(f: wp.array2d(dtype=wp.vec2), x: int, y: int):
    if x < 0 or x >= grid_width:
        return wp.vec2()
    if y < 0 or y >= grid_height:
        return wp.vec2()

    return f[x, y]


@wp.func
def sample_vel(f: wp.array2d(dtype=wp.vec2), x: float, y: float):
    lx = int(wp.floor(x))
    ly = int(wp.floor(y))

    tx = x - float(lx)
    ty = y - float(ly)

    s0 = wp.lerp(lookup_vel(f, lx, ly), lookup_vel(f, lx + 1, ly), tx)
    s1 = wp.lerp(lookup_vel(f, lx, ly + 1), lookup_vel(f, lx + 1, ly + 1), tx)

    s = wp.lerp(s0, s1, ty)
    return s