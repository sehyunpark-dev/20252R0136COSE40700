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

@wp.kernel
def advect(
    u0: wp.array2d(dtype=wp.vec2),
    u1: wp.array2d(dtype=wp.vec2),
    rho0: wp.array2d(dtype=float),
    rho1: wp.array2d(dtype=float),
    dt: float,
):
    i, j = wp.tid()

    u = u0[i, j]

    # trace backward
    p = wp.vec2(float(i), float(j))
    p = p - u * dt

    # advect
    u1[i, j] = sample_vel(u0, p[0], p[1])
    rho1[i, j] = sample_float(rho0, p[0], p[1])


@wp.kernel
def divergence(u: wp.array2d(dtype=wp.vec2), div: wp.array2d(dtype=float)):
    i, j = wp.tid()

    if i == grid_width - 1:
        return
    if j == grid_height - 1:
        return

    dx = (u[i + 1, j][0] - u[i, j][0]) * 0.5
    dy = (u[i, j + 1][1] - u[i, j][1]) * 0.5

    div[i, j] = dx + dy

@wp.kernel
def pressure_solve(p0: wp.array2d(dtype=float), p1: wp.array2d(dtype=float), div: wp.array2d(dtype=float)):
    i, j = wp.tid()

    s1 = lookup_float(p0, i - 1, j)
    s2 = lookup_float(p0, i + 1, j)
    s3 = lookup_float(p0, i, j - 1)
    s4 = lookup_float(p0, i, j + 1)

    # Jacobi update
    err = s1 + s2 + s3 + s4 - div[i, j]

    p1[i, j] = err * 0.25


@wp.kernel
def pressure_apply(p: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2)):
    i, j = wp.tid()

    if i == 0 or i == grid_width - 1:
        return
    if j == 0 or j == grid_height - 1:
        return

    # pressure gradient
    f_p = wp.vec2(p[i + 1, j] - p[i - 1, j], p[i, j + 1] - p[i, j - 1]) * 0.5

    u[i, j] = u[i, j] - f_p