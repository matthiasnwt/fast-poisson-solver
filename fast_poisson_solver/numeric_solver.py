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
#
# This module contains the numeric solver for the PDEs.
# The code is taken from Mohammad Asif Zaman, https://github.com/zaman13/Poisson-solver-2D.
# See their publication:
# Zaman, M.A. "Numerical Solution of the Poisson Equation Using Finite Difference Matrix Operators",
# Electronics 2022, 11, 2365. https://doi.org/10.3390/electronics11152365


import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import spsolve

from .diff_matrices import Diff_mat_2D
from .utils import format_input

warnings.simplefilter('ignore', SparseEfficiencyWarning)


def numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc, precision=torch.float32, verbose=0):
    """
    This function numerically solves the partial differential equation (PDE) using the provided source function,
    PDE coordinates, boundary condition, and boundary condition coordinates. It uses the precision specified,
    measures the time taken for the operation if requested, and controls the verbosity of the output.

    Parameters
    ----------
    f : tensor/array/list
        The source function for the PDE.
    x_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    y_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    u_bc : tensor/array/list
        The boundary condition for the PDE.
    x_bc : tensor/array/list
        Coordinates of the boundary condition.
    y_bc : tensor/array/list
        Coordinates of the boundary condition.
    precision : torch.dtype, optional
        The precision to be used for the numeric solver. Default is torch.float32.
    verbose : int, optional
        Controls the verbosity of the output. If 0, only the solution 'u' is returned. If greater than 0,
        both the solution 'u' and runtime 'delta t' are returned. Default is 1.

    Returns
    -------
    tuple
        u : tensor
            The complete numeric solution of the PDE.

        t : float
            The runtime, i.e., the time it took the method to run in seconds.

    References
    ----------
    Zaman, M.A. "Numerical Solution of the Poisson Equation Using Finite Difference Matrix Operators",
    Electronics 2022, 11, 2365. https://doi.org/10.3390/electronics11152365

    See also: https://github.com/zaman13/Poisson-solver-2D
    """
    f, x_pde, y_pde, u_bc, x_bc, y_bc = format_input([f, x_pde, y_pde, u_bc, x_bc, y_bc],
                                                           precision=precision, device="cpu", as_array=True)

    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        # add more mappings as needed
    }

    precision = dtype_map.get(precision, None)

    x = np.concatenate([x_pde, x_bc], axis=0)
    y = np.concatenate([y_pde, y_bc], axis=0)

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    Nx = len(x_unique)
    Ny = len(y_unique)

    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]

    Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx, Ny)
    Dx_2d = Dx_2d.astype(precision)
    Dy_2d = Dy_2d.astype(precision)
    D2x_2d = D2x_2d.astype(precision)
    D2y_2d = D2y_2d.astype(precision)

    # Construction of the system matrix and adjust the right hand vector for boundary conditions
    I_sp = sp.eye(Nx * Ny).tocsr().astype(precision)
    L_sys = D2x_2d / dx ** 2 + D2y_2d / dy ** 2

    BD = I_sp  # .tolil()  # Dirichlet boundary operator
    BNx = Dx_2d  # .tolil()  # Neumann boundary operator for x component
    BNy = Dy_2d  # .tolil()  # Neumann boundary operator for y component

    t0_run = time.perf_counter()

    xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    ind = np.lexsort((xy[:, 0], xy[:, 1]))
    reverse_ind = np.argsort(ind)

    b_ind = [np.argwhere(ind >= len(x_pde)).reshape(-1)]

    f = np.concatenate([f, u_bc], axis=0)
    f = f[ind]

    b_type = [0]
    b_val = [u_bc[0]]

    N_B = len(b_val)

    # Selectively replace the rows of the system matrix that correspond to
    # boundary value points. We replace these rows with
    # those of the boundary operator

    for m in range(N_B):
        # print(f[b_ind[m]], b_val[m] )
        # f[b_ind[m]] = b_val[m]  # Insert boundary values at the outer boundary points
        if b_type[m] == 0:
            L_sys[b_ind[m], :] = BD[b_ind[m], :]
        elif b_type[m] == 1:
            L_sys[b_ind[m], :] = BNx[b_ind[m], :]
        elif b_type[m] == 2:
            L_sys[b_ind[m], :] = BNy[b_ind[m], :]


    u = spsolve(L_sys, f)

    u = u[reverse_ind] # reorder the solution to the original order
    u = u.astype(precision)
    end = time.perf_counter()

    dt = end - t0_run

    if verbose > 0:
        print(f"Time Numeric Solver: {dt:.6f} s")

    return u, dt


if __name__ == "__main__":
    n_x = n_y = 100
    domain_x = [0, 1]
    domain_y = [0, 1]
    k = 1.5

    x = np.linspace(domain_x[0], domain_x[1], n_x)
    y = np.linspace(domain_y[0], domain_y[1], n_y)
    xx, yy = np.meshgrid(x, y)
    x = xx.flatten()
    y = yy.flatten()

    f = np.sin(k * np.pi * x) * np.sin(k * np.pi * y) * (np.pi ** 2)

    b_ind = [np.concatenate(
        (np.arange(n_x), np.arange(n_x) + n_x * (n_y - 1), np.arange(n_y) * n_x, np.arange(n_y) * n_x + n_x - 1))]

    b_type = [0]
    b_val = [0]

    # solver = Solver(x, y, f, b_ind, b_type, b_val)
    # solver.solve()

    u = numeric_solve(x, y, f, b_ind, b_type, b_val)
    plt.imshow(u)
    plt.savefig("u3.png")
