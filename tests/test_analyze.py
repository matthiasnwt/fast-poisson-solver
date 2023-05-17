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

from fast_poisson_solver import Data, numeric_solve, Solver, analyze
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

    def test1_analyze_full(self):
        res = analyze(self.f_ml, self.f, self.u_ml_bc, self.u_bc, self.u_ml, self.u_num, normalize=False)
        self.assertTrue('f' in res)
        self.assertTrue('bc' in res)
        self.assertTrue('u' in res)
        self.assertTrue('MSE' in res['f'])
        self.assertTrue('RMSE' in res['f'])
        self.assertTrue('MAE' in res['f'])
        self.assertTrue('rMAE' in res['f'])
        self.assertTrue('SSIM' in res['f'])
        self.assertTrue('PSNR' in res['f'])
        self.assertTrue('R2' in res['f'])
        self.assertTrue('MSE' in res['u'])
        self.assertTrue('RMSE' in res['u'])
        self.assertTrue('MAE' in res['u'])
        self.assertTrue('rMAE' in res['u'])
        self.assertTrue('SSIM' in res['u'])
        self.assertTrue('PSNR' in res['u'])
        self.assertTrue('R2' in res['u'])
        self.assertTrue('MSE' in res['bc'])
        self.assertTrue('RMSE' in res['bc'])
        self.assertTrue('MAE' in res['bc'])
        self.assertTrue('rMAE' in res['bc'])

    def test2_analyze_f_bc(self):
        res = analyze(self.f_ml, self.f, self.u_ml_bc, self.u_bc, normalize=False)
        self.assertTrue('f' in res)
        self.assertTrue('bc' in res)
        self.assertFalse('u' in res)

    def test3_analyze_normalize(self):
        _ = analyze(self.f_ml, self.f, self.u_ml_bc, self.u_bc, self.u_ml, self.u_num, normalize=True)


if __name__ == '__main__':
    unittest.main()