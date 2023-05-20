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
from fast_poisson_solver import Solver, Data, numeric_solve, plot_comparison, plot, analyze, plot_side_by_side, bicubic_interpolate


device = torch.device('cuda')

grid_size = 32
name = 'geo'
save_path = '../assets'
seed = 0
lambdas_pde = [2 ** -12]

cases = [{'name': 'perlin', 'param': 'random', 'b_val': 'random'}]

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

u_num, t_numeric = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc,verbose=1, precision=torch.float32)

solver = Solver(device=device, verbose=1, precision=torch.float32, use_weights=True, lambdas_pde=lambdas_pde)
solver.precompute(x_pde, y_pde, x_bc, y_bc, name='200', save=False, load=False)
u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)



# plot_side_by_side(x_pde, y_pde, x_bc, y_bc, u_ml, f, f_ml, u_num, grid=True, show=True)

# u_ml = bicubic_interpolate(x_pde, y_pde, x_bc, y_bc, u_ml, x_pde_num, y_pde_num, x_bc_num, y_bc_num)
# f_ml = bicubic_interpolate(x_pde, y_pde, x_bc, y_bc, f_ml, x_pde_num, y_pde_num, x_bc_num, y_bc_num, domain=True)

# res = analyze(f_ml, f, u_ml_bc, u_bc, normalize=True)
# print(res)
# solver.plot_lambda_error()
plot_comparison(x_pde, y_pde, x_bc, y_bc, u_ml, f, f_ml, u_num, grid=True, show=True)
# plot(x_pde, y_pde, x_bc, y_bc, u_ml, f, f_ml, grid=False, show=True)
# solver.plot_w(show=True)
# solver.plot_H(show=False)


