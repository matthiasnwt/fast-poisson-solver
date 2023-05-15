# Copyright (C) 2023 Matthias Neuwirth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.



from random import random

import numpy as np
from perlin_noise import PerlinNoise


def sincos(x, y, params):
    def generate_random_params(num_terms):
        k = np.random.uniform(-10, 10, (2, num_terms))
        offsets = np.random.uniform(-1, 1, (2, num_terms))
        s = np.random.uniform(-100, 100, num_terms)
        trig_funcs = np.random.choice(['sin', 'cos'], num_terms)
        return k, offsets, s, trig_funcs

    if params.startswith('random'):
        params_splitted = params.split('_')
        if len(params_splitted) > 1:
            params = params_splitted[0]
            scale_global = float(params_splitted[1])
        else:
            scale_global = None

    if params == 'random1':
        num_terms = 1
        k, offsets, s, trig_funcs = generate_random_params(num_terms)
    elif params == 'random5':
        num_terms = 5
        k, offsets, s, trig_funcs = generate_random_params(num_terms)
    elif params == 'random10':
        num_terms = 10
        k, offsets, s, trig_funcs = generate_random_params(num_terms)
    elif params == 'random':
        num_terms = np.random.randint(1, 10)
        k, offsets, s, trig_funcs = generate_random_params(num_terms)
    else:
        num_terms = len(params[0][0])
        k, offsets, s, trig_funcs = params

    img = np.zeros_like(x)

    for i in range(num_terms):
        k1, k2 = k[:, i]
        x_off, y_off = offsets[:, i]
        s_i = s[i]
        trig_func = trig_funcs[i]

        if trig_func == 'sin':
            img += np.sin(k1 * np.pi * (x - x_off)) * np.sin(k2 * np.pi * (y - y_off)) * s_i
        else:  # 'cos'
            img += np.sin(k1 * np.pi * (x - x_off)) * np.cos(k2 * np.pi * (y - y_off)) * s_i

    if scale_global is not None:
        max_img, min_img = np.max(img), np.min(img)
        img = (img - min_img) / (max_img - min_img) * scale_global

    return img


# def rectangle(x, y, params):
#     if params == 'random':
#         num = np.random.randint(1, 10)
#         l_x, l_y = np.random.uniform(2/self.grid_num, 1/2, (2, num))
#         x_c, y_c = np.random.uniform(0, 1, (2, num))
#         s = np.random.uniform(-100, 100, num)
#     else:
#         (l_x, l_y), (x_c, y_c), s = params
#         num = len(l_x)
#
#     img = np.zeros_like(x)
#     for i in range(num):
#         img[(x >= x_c[i] - l_x[i] / 2) & (x <= x_c[i] + l_x[i] / 2) & (y >= y_c[i] - l_y[i] / 2) & (
#                     y <= y_c[i] + l_y[i] / 2)] += s[i]
#
#     return img
#
# def circle(self, x, y, params):
#     if params == 'random':
#         num = np.random.randint(1, 10)
#         r = np.random.uniform(2/self.grid_num, 0.5, num)
#         x_c, y_c = np.random.uniform(0, 1, (2, num))
#         s = np.random.uniform(-100, 100, num)
#     else:
#         r, (x_c, y_c), s = params
#         num = len(r)
#
#     img = np.zeros_like(x)
#     for i in range(num):
#         d = np.sqrt((x - x_c[i]) ** 2 + (y - y_c[i]) ** 2)
#         img[d <= r[i]] += s[i]
#
#     return img

def rectangle_circle(x, y, params, grid_num):
    if params.startswith('random'):
        params_splitted = params.split('_')
        if len(params_splitted) > 1:
            params = params_splitted[0]
            scale_global = float(params_splitted[1])
        else:
            scale_global = None

    if params == 'random1':
        r = np.random.binomial(size=1, n=1, p=0.5)
    elif params == 'random5':
        r = np.random.binomial(size=5, n=1, p=0.5)
    elif params == 'random10':
        r = np.random.binomial(size=10, n=1, p=0.5)
    elif params == 'random':
        num = np.random.randint(1, 10)
        r = np.random.binomial(size=num, n=1, p=0.5)
    else:
        (r_circ, (x_c_circ, y_c_circ), s_circ), ((l_x, l_y), (x_c_rect, y_c_rect), s_rect) = params
        r = [1] * len(r_circ) + [0] * len(l_x)

    num_circ = np.sum(r)
    num_rect = len(r) - num_circ

    if params.startswith('random'):
        l_x, l_y = np.random.uniform(2 / grid_num, 1 / 2, (2, num_rect))
        x_c_rect, y_c_rect = np.random.uniform(0, 1, (2, num_rect))
        s_rect = np.random.uniform(-100, 100, num_rect)

        r_circ = np.random.uniform(2 / grid_num, 1 / 2, num_circ)
        x_c_circ, y_c_circ = np.random.uniform(0, 1, (2, num_circ))
        s_circ = np.random.uniform(-100, 100, num_circ)

    img = np.zeros_like(x)

    # Rectangles
    for i in range(num_rect):
        img[(x >= x_c_rect[i] - l_x[i] / 2) & (x <= x_c_rect[i] + l_x[i] / 2) & (y >= y_c_rect[i] - l_y[i] / 2) & (
                y <= y_c_rect[i] + l_y[i] / 2)] += s_rect[i]

    # Circles
    for i in range(num_circ):
        d = np.sqrt((x - x_c_circ[i]) ** 2 + (y - y_c_circ[i]) ** 2)
        img[d <= r_circ[i]] += s_circ[i]

    if scale_global is not None:
        max_img, min_img = np.max(img), np.min(img)
        img = (img - min_img) / (max_img - min_img) * scale_global

    return img


def exp(x, y, params):
    if params == 'random':
        s = np.random.uniform(-100, 100)
        x_off, y_off = np.random.uniform(-1, 1, 2)
    else:
        (x_off, y_off), s = params
    img = np.exp(-(x - x_off) ** 2 - (y - y_off) ** 2) * s
    return img


def perlin(x, y, params):
    randomscale = True
    scale_global = None
    if isinstance(params, str):
        if params.startswith('random'):
            params_splitted = params.split('_')
            if len(params_splitted) > 1:
                params = params_splitted[0]
                scale_global = float(params_splitted[1])

        if params.endswith('-'):
            randomscale = False
            params = params[:-1]
        else:
            randomscale = True

    if params == 'random1':
        num = 1
    elif params == 'random5':
        num = 5
    elif params == 'random10':
        num = 10
    elif params == 'random':
        num = np.random.randint(1, 6)
    else:
        octaves, seeds, s = params

    if isinstance(params, str):
        if params.startswith('random'):
            s = np.random.uniform(-100, 100, num)
            octaves = np.random.uniform(0.1, 12, num)
            seeds = np.random.randint(1, 1000000, num)

    # print(num, octaves, seeds, s)

    u = np.zeros_like(x)
    for octave, seed, si in zip(octaves, seeds, s):
        noise = PerlinNoise(octaves=octave, seed=seed)
        u += np.array([noise([x, y]) for x, y in zip(x, y)]) * si

    if randomscale:
        if np.random.random() > 0.3:
            s = np.random.uniform(-1, 1)
            octave = np.random.uniform(0.1, 3)
            seed = np.random.randint(1, 1000000)

            noise = PerlinNoise(octaves=octave, seed=seed)
            u *= np.array([noise([x, y]) for x, y in zip(x, y)]) * s

    if scale_global is not None:
        max_img, min_img = np.max(u), np.min(u)
        u = (u - min_img) / (max_img - min_img) * scale_global - scale_global / 2 + np.random.uniform(- scale_global / 2,
                                                                                                   scale_global / 2)

    return u