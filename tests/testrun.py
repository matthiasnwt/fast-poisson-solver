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


import torch
from fast_poisson_solver import Solver, Data, numeric_solve, plot_comparison, plot, analyze
from fast_poisson_solver.utils import bicubic_interpolate

device = torch.device('cuda:1')

grid_size = 32
name = 'test'
save_path = '../Auswertungen/Test'
seed = 1
lambdas_pde = [2 ** -11]

cases = [{'name': 'perlin', 'param': 'random', 'b_val': 0}]

data = Data(domain_x=[0, 1],
            domain_y=[0, 1],
            grid_num=grid_size,
            random_coords=False,
            cases=cases,
            noise_std=0,
            use_torch=True,
            device=device,
            precision=torch.float32,
            seed=seed)

f_num, x_pde_num, y_pde_num, u_bc, x_bc_num, y_bc_num = data.__call__(0)

u_num, t_numeric = numeric_solve(f_num, x_pde_num, y_pde_num, u_bc, x_bc_num, y_bc_num, timeit=True, verbose=1, precision=torch.float32)

data = Data(domain_x=[0, 1],
            domain_y=[0, 1],
            grid_num=grid_size,
            random_coords=False,
            cases=cases,
            noise_std=0,
            use_torch=True,
            device=device,
            precision=torch.float32,
            seed=seed)

f, x_pde, y_pde, u_bc, x_bc, y_bc = data.__call__(0)

# u_num, t_numeric = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc, timeit=True, verbose=1, precision=torch.float32)

solver = Solver(device=device, verbose=1, precision=torch.float32, use_weights=True)
solver.precompute(x_pde, y_pde, x_bc, y_bc, name='32', save=True, load=False)
u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)




# u_ml = bicubic_interpolate(x_pde, y_pde, x_bc, y_bc, u_ml, x_pde_num, y_pde_num, x_bc_num, y_bc_num)
# f_ml = bicubic_interpolate(x_pde, y_pde, x_bc, y_bc, f_ml, x_pde_num, y_pde_num, x_bc_num, y_bc_num, domain=True)

res = analyze(f_ml, f, u_ml_bc, u_bc, normalize=True)
print(res)
# solver.plot_lambda_error()
# plot_comparison(x_pde_num, y_pde_num, x_bc_num, y_bc_num, u_ml, f_num, f_ml, u_num, grid=True, show=True)
# plot(x_pde, y_pde, x_bc, y_bc, u_ml, f, f_ml, grid=False, show=True)
# solver.plot_w(show=True)
# solver.plot_H(show=False)
