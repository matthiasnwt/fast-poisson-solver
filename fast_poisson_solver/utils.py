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



import numpy as np
import torch
from scipy.interpolate import interpolate, griddata
from torch.autograd import grad

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def sumsumAB(A, B=None):
    precision = A.dtype
    if B is None:
        B = A
    device = A.device
    A_N = A.shape[0]
    B_N = B.shape[0]
    ones_A = torch.ones(A_N, 1, device=device).to(precision)
    ones_B = torch.ones(B_N, 1, device=device).to(precision)
    mm_A = torch.mm(A.t(), ones_A)
    mm_B = torch.mm(B.t(), ones_B)
    mm = torch.mm(mm_A, mm_B.t())
    return mm

def sumsumAb(A, b):
    precision = A.dtype
    device = A.device
    A_N = A.shape[0]
    b_N = b.shape[0]
    ones_A = torch.ones(A_N, 1, device=device).to(precision)
    ones_b = torch.ones(b_N, 1, device=device).to(precision)
    mm_A = torch.mm(A.t(), ones_A)
    mm_b = torch.mm(b.t(), ones_b)
    mm = torch.mm(mm_A, mm_b)
    return mm

def calculate_laplace(u, x, y):
    laplace = []
    for i in range(u.shape[1]):
        u_i = u[:, [i]]
        u_x = grad(u_i, x, create_graph=True, grad_outputs=torch.ones_like(u_i), retain_graph=True)[0]
        u_y = grad(u_i, y, create_graph=True, grad_outputs=torch.ones_like(u_i), retain_graph=True)[0]
        u_xx = grad(u_x, x, create_graph=False, grad_outputs=torch.ones_like(u_x), retain_graph=True)[
            0]  # .detach()
        u_yy = grad(u_y, y, create_graph=False, grad_outputs=torch.ones_like(u_y), retain_graph=True)[
            0]  # .detach()
        laplace_i = u_xx + u_yy
        laplace.append(laplace_i)

    laplace = torch.cat(laplace, dim=1)
    return laplace

def shape_and_size(name, tensor):
    size = tensor.element_size() * tensor.nelement()
    shape = list(tensor.shape)
    print(f'{name}: Shape: {shape} Size: {size} Bytes')

def sort_ascend(x, y, v, return_index=False):
    xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    ind = np.lexsort((xy[:, 0], xy[:, 1]))
    x = x[ind]
    y = y[ind]
    v = [v_i[ind] for v_i in v]
    if return_index:
        return x, y, v, ind
    else:
        return x, y, v

def minmax(v1, v2=None):
    if v2 is None:
        v2 = v1
    vmin = min((np.min(v1), np.min(v2)))
    vmax = max((np.max(v1), np.max(v2)))
    return vmin, vmax

def format_input(v, precision=torch.float32, device='cpu', as_array=False, reshape=True):
    if isinstance(device, str):
        device = torch.device(device)
    for i in range(len(v)):
        if isinstance(v[i], list):
            v[i] = torch.tensor(v[i], dtype=precision, device=device)
        elif isinstance(v[i], np.ndarray):
            v[i] = torch.tensor(v[i], dtype=precision, device=device)
        elif isinstance(v[i], int) or isinstance(v[i], float):
            continue
        else:
            v[i] = v[i].to(precision).to(device)

    if reshape:
        v = [v_i.reshape(-1, 1) for v_i in v]
    if as_array:
        v = [v_i.detach().cpu().numpy() for v_i in v]
    return v


def process_grid_data(x, y, z1, z2=None):
    x_num = len(np.unique(x))
    y_num = len(np.unique(y))

    if z2 is not None:
        x, y, [z1, z2] = sort_ascend(x, y, [z1, z2])
        z1 = z1.reshape(x_num, y_num)
        z2 = z2.reshape(x_num, y_num)
        return x, y, z1, z2
    else:
        x, y, [z1] = sort_ascend(x, y, [z1])
        z1 = z1.reshape(x_num, y_num)
        return x, y, z1


def bicubic_interpolate(x_pde_base, y_pde_base, x_bc_base, y_bc_base, v_base, x_pde_new, y_pde_new, x_bc_new, y_bc_new,
                        domain=False):
    """
    Interpolates values for new x, y coordinates using bicubic interpolation

    :param x: numpy array, original x coordinates
    :param y: numpy array, original y coordinates
    :param v: numpy array, values at original coordinates
    :param x_new: numpy array, new x coordinates to predict values for
    :param y_new: numpy array, new y coordinates to predict values for
    :return: numpy array, interpolated values at new coordinates
    """
    # First, we'll need to create a grid of x, y values

    x_pde_base, y_pde_base, x_bc_base, y_bc_base, v_base, x_pde_new, y_pde_new, x_bc_new, y_bc_new = format_input(
        [x_pde_base, y_pde_base, x_bc_base, y_bc_base, v_base, x_pde_new, y_pde_new, x_bc_new, y_bc_new],
        precision=torch.float64, device='cpu', as_array=True, reshape=True)

    if domain:
        x_base = x_pde_base.reshape(-1)
        y_base = y_pde_base.reshape(-1)
        x_new = x_pde_new.reshape(-1)
        y_new = y_pde_new.reshape(-1)
    else:
        x_base = np.concatenate([x_pde_base, x_bc_base]).reshape(-1)
        y_base = np.concatenate([y_pde_base, y_bc_base]).reshape(-1)
        x_new = np.concatenate([x_pde_new, x_bc_new]).reshape(-1)
        y_new = np.concatenate([y_pde_new, y_bc_new]).reshape(-1)

    coordinates_base = np.array([x_base, y_base]).T
    coordinates_new = np.array([x_new, y_new]).T

    v_new = griddata(coordinates_base, v_base.reshape(-1), coordinates_new, method='cubic')

    return v_new


if __name__ == '__main__':

    A = torch.rand(3, 2)
    B = torch.rand(4, 1)
    mm = sumsumAb(A, B)
    print(mm)