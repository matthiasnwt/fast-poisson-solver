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
        self.device_cuda = torch.device('cuda')
        self.device_cpu = torch.device('cpu')
        self.float32 = torch.float32
        self.float64 = torch.float64

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
                                 device=self.device_cuda,
                                 precision=self.float32,
                                 random_coords=False,
                                 seed=0)
        self.f, self.x_pde, self.y_pde, self.u_bc, self.x_bc, self.y_bc = data.__call__(0)

    def test_instantiation_base(self):
        solver = Solver()
        self.assertTrue(isinstance(solver, Solver))

    def test_instantiation_cuda(self):
        solver = Solver(device=self.device_cuda)
        self.assertTrue(isinstance(solver, Solver))
        self.assertTrue(solver.device == self.device_cuda)

    def test_instantiation_cpu(self):
        solver = Solver(device=self.device_cpu)
        self.assertTrue(isinstance(solver, Solver))
        self.assertTrue(solver.device == self.device_cpu)

    def test_instantiation_float32(self):
        solver = Solver(precision=self.float32)
        self.assertTrue(isinstance(solver, Solver))
        self.assertTrue(solver.precision == self.float32)

    def test_instantiation_float64(self):
        solver = Solver(precision=self.float64)
        self.assertTrue(isinstance(solver, Solver))
        self.assertTrue(solver.precision == self.float64)

    def test_instantiation_no_weights(self):
        solver = Solver(use_weights=False)
        self.assertTrue(isinstance(solver, Solver))

    def test_instantiation_no_compiling(self):
        solver = Solver(compile_model=False)
        self.assertTrue(isinstance(solver, Solver))

    def test_precompute_base(self):
        solver = Solver()
        solver.precompute(self.x_pde, self.y_pde, self.x_bc, self.y_bc)

    def test_precompute_save(self):
        solver = Solver()
        solver.precompute(self.x_pde, self.y_pde, self.x_bc, self.y_bc, name='test', save=True)

    def test_precompute_load(self):
        solver = Solver()
        solver.precompute(self.x_pde, self.y_pde, self.x_bc, self.y_bc, name='test', load=True)

if __name__ == '__main__':
    unittest.main()
