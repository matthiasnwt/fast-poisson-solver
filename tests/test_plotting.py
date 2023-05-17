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

import unittest

from fast_poisson_solver import Data, numeric_solve, Solver, plot, plot_comparison, plot_side_by_side
import torch

class Test(unittest.TestCase):

    def setUp(self):
        data = Data(domain_x=[0, 1],
                    domain_y=[0, 1],
                    grid_num=5,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=0,
                    shuffle=False,
                    initial_shuffle=False,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=True,
                    device=torch.device('cuda'),
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        self.f, self.x_pde, self.y_pde, self.u_bc, self.x_bc, self.y_bc = data.__call__(0)
        self.solver = Solver()
        self.solver.precompute(self.x_pde, self.y_pde, self.x_bc, self.y_bc)
        self.u_ml, self.u_ml_pde, self.u_ml_bc, self.f_ml, _ = self.solver.solve(self.f, self.u_bc)
        self.u_num, _ = numeric_solve(self.f, self.x_pde, self.y_pde, self.u_bc, self.x_bc, self.y_bc)

    def test1_plot_non_grid(self):
        plot(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.u_ml, self.f, self.f_ml, grid=False, show=False)

    def test2_plot_grid(self):
        plot(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.u_ml, self.f, self.f_ml, grid=True, show=False)

    def test3_plot_comparison_non_grid(self):
        plot_comparison(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.u_ml, self.f, self.f_ml, self.u_num,
                        grid=False, show=False)

    def test4_plot_comparison_grid(self):
        plot_comparison(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.u_ml, self.f, self.f_ml, self.u_num,
                        grid=True, show=False)

    def test5_plot_side_by_side_non_grid(self):
        plot_side_by_side(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.u_ml, self.f, self.f_ml, self.u_num,
                          grid=False, show=False)

    def test6_plot_side_by_side_grid(self):
        plot_side_by_side(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.u_ml, self.f, self.f_ml, self.u_num,
                          grid=True, show=False)


if __name__ == '__main__':
    unittest.main()