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

from fast_poisson_solver import Data, numeric_solve
import torch
import io
import contextlib

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

    def test1_numeric_base(self):
        _ = numeric_solve(self.f, self.x_pde, self.y_pde, self.u_bc, self.x_bc, self.y_bc)

    def test2_numeric_float64(self):
        _ = numeric_solve(self.f, self.x_pde, self.y_pde, self.u_bc, self.x_bc, self.y_bc, precision=torch.float64)

    def test3_numeric_verbose(self):
        expected_start = "Time Numeric Solver:"
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            _ = numeric_solve(self.f, self.x_pde, self.y_pde, self.u_bc, self.x_bc, self.y_bc, verbose=1)
        actual_output = captured_output.getvalue()
        self.assertTrue(actual_output.startswith(expected_start))

if __name__ == '__main__':
    unittest.main()

