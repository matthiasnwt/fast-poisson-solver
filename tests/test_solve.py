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

from fast_poisson_solver import Solver, Data, numeric_solve
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

    def test_solve(self):
        _ = self.solver.solve(self.f, self.u_bc)


if __name__ == '__main__':
    unittest.main()
