"""
Copyright (C) 2023 Matthias Neuwirth

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import unittest

import numpy as np

from fast_poisson_solver import Data
import torch




class Test(unittest.TestCase):

    def setUp(self):
        self.device_cpu = torch.device('cpu')
        self.device_cuda = torch.device('cuda')

        self.data_default = Data(domain_x=[0, 1],
                                 domain_y=[0, 1],
                                 grid_num=5,
                                 cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                                 noise_std=0,
                                 shuffle=False,
                                 initial_shuffle=False,
                                 batchsize=-1,
                                 batchsize_bc=-1,
                                 use_torch=True,
                                 device=self.device_cpu,
                                 precision=torch.float32,
                                 random_coords=False,
                                 seed=0)


    def test_data_numpy(self):
        data = Data(domain_x=[0, 1],
                    domain_y=[0, 1],
                    grid_num=5,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=0,
                    shuffle=False,
                    initial_shuffle=False,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=False,
                    device=self.device_cpu,
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        f, x_pde, y_pde, u_bc, x_bc, y_bc = data.__call__(0)

        self.assertIsInstance(f, np.ndarray)
        self.assertIsInstance(x_pde, np.ndarray)
        self.assertIsInstance(y_pde, np.ndarray)
        self.assertIsInstance(u_bc, np.ndarray)
        self.assertIsInstance(x_bc, np.ndarray)
        self.assertIsInstance(y_bc, np.ndarray)

        self.assertEqual(f.shape, x_pde.shape)
        self.assertEqual(x_pde.shape, y_pde.shape)
        self.assertEqual(u_bc.shape, x_bc.shape)
        self.assertEqual(x_bc.shape, y_bc.shape)

    def test_data_cpu(self):
        f, x_pde, y_pde, u_bc, x_bc, y_bc = self.data_default.__call__(0)

        self.assertIsInstance(f, torch.Tensor)
        self.assertIsInstance(x_pde, torch.Tensor)
        self.assertIsInstance(y_pde, torch.Tensor)
        self.assertIsInstance(u_bc, torch.Tensor)
        self.assertIsInstance(x_bc, torch.Tensor)
        self.assertIsInstance(y_bc, torch.Tensor)

        self.assertEqual(f.dtype, torch.float32)
        self.assertEqual(x_pde.dtype, torch.float32)
        self.assertEqual(y_pde.dtype, torch.float32)
        self.assertEqual(u_bc.dtype, torch.float32)
        self.assertEqual(x_bc.dtype, torch.float32)
        self.assertEqual(y_bc.dtype, torch.float32)

    def test_data_cuda(self):
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
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        f, x_pde, y_pde, u_bc, x_bc, y_bc = data.__call__(0)

        self.assertIsInstance(f, torch.Tensor)
        self.assertIsInstance(x_pde, torch.Tensor)
        self.assertIsInstance(y_pde, torch.Tensor)
        self.assertIsInstance(u_bc, torch.Tensor)
        self.assertIsInstance(x_bc, torch.Tensor)
        self.assertIsInstance(y_bc, torch.Tensor)

    def test_data_cases(self):
        cases = [
            {'name': 'perlin', 'param': 'random1', 'b_val': 'random'},
            {'name': 'perlin', 'param': 'random10', 'b_val': 'random'},
            {'name': 'perlin', 'param': 'random', 'b_val': 'random'},
            {'name': 'sin', 'param': 'random1', 'b_val': 'random'},
            {'name': 'sin', 'param': 'random10', 'b_val': 'random'},
            {'name': 'sin', 'param': 'random', 'b_val': 'random'},
            {'name': 'geo', 'param': 'random1', 'b_val': 'random'},
            {'name': 'geo', 'param': 'random10', 'b_val': 'random'},
            {'name': 'geo', 'param': 'random', 'b_val': 'random'},
        ]
        for i, case in enumerate(cases):
            data = Data(domain_x=[0, 1],
                        domain_y=[0, 1],
                        grid_num=5,
                        cases=[case],
                        noise_std=0,
                        shuffle=False,
                        initial_shuffle=False,
                        batchsize=-1,
                        batchsize_bc=-1,
                        use_torch=True,
                        device=self.device_cpu,
                        precision=torch.float32,
                        random_coords=False,
                        seed=0)

            _ = data.__call__(0)

    def test_data_float64(self):
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
                    device=self.device_cpu,
                    precision=torch.float64,
                    random_coords=False,
                    seed=0)

        f, x_pde, y_pde, u_bc, x_bc, y_bc = data.__call__(0)

        self.assertEqual(f.dtype, torch.float64)
        self.assertEqual(x_pde.dtype, torch.float64)
        self.assertEqual(y_pde.dtype, torch.float64)
        self.assertEqual(u_bc.dtype, torch.float64)
        self.assertEqual(x_bc.dtype, torch.float64)
        self.assertEqual(y_bc.dtype, torch.float64)

    def test_data_shuffle(self):
        data = Data(domain_x=[0, 1],
                    domain_y=[0, 1],
                    grid_num=5,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=0,
                    shuffle=True,
                    initial_shuffle=False,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=True,
                    device=self.device_cpu,
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        _ = data.__call__(0)

    def test_data_initial_shuffle(self):
        data = Data(domain_x=[0, 1],
                    domain_y=[0, 1],
                    grid_num=5,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=0,
                    shuffle=False,
                    initial_shuffle=True,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=True,
                    device=self.device_cpu,
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        _ = data.__call__(0)

    def test_data_noise(self):
        data = Data(domain_x=[0, 1],
                    domain_y=[0, 1],
                    grid_num=5,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=1,
                    shuffle=False,
                    initial_shuffle=False,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=True,
                    device=self.device_cpu,
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        _ = data.__call__(0)

    def test_data_random_coords(self):
        data = Data(domain_x=[0, 1],
                    domain_y=[0, 1],
                    grid_num=2,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=0,
                    shuffle=False,
                    initial_shuffle=False,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=True,
                    device=self.device_cpu,
                    precision=torch.float32,
                    random_coords=True,
                    seed=0)

        _ = data.__call__(0)

    def test_data_domain_size(self):
        data = Data(domain_x=[0.1, 0.3],
                    domain_y=[0.4, 0.9],
                    grid_num=5,
                    cases=[{'name': 'perlin', 'param': 'random', 'b_val': 'random'}],
                    noise_std=0,
                    shuffle=False,
                    initial_shuffle=False,
                    batchsize=-1,
                    batchsize_bc=-1,
                    use_torch=True,
                    device=self.device_cpu,
                    precision=torch.float32,
                    random_coords=False,
                    seed=0)

        _ = data.__call__(0)

    #
    # def test_numeric(self):
    #     f, x_pde, y_pde, u_bc, x_bc, y_bc = self.data_default.__call__(0)
    #     _ = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc, timeit=False, verbose=0)
    #
    # def test_precompute(self):
    #     _, x_pde, y_pde, _, x_bc, y_bc = self.data_default.__call__(0)
    #     interfer = Solver(device=self.device_cpu, verbose=1, use_weights=True)
    #     interfer.precompute(x_pde, y_pde, x_bc, y_bc, save=False, load=False)

    # if ratio == 1 and noise_std == 0:
    #     u_ml, u_ml_pde, u_ml_bc, t_ml = interfer.run(f, u_bc)
    # else:
    #     u_ml, u_ml_pde, u_ml_bc, t_ml = interfer.run_limited_data(f, u_bc, ratio, ratio, noise_std=noise_std)
    # res = interfer.analyze(u_numeric, u_bc, normalize=True)
    # print(res)
    #
    # # interfer.plot_lambda_error()
    # interfer.plot(u_numeric, save_path, show=True)
    # # interfer.plot_w(save_path, show=True)
    # # interfer.plot_H(save_path, show=False)


#    def test_run(self):
#         A = TwoDimensional(...)
# A.compile()
# A.run()
# Add assertions to check the expected results

if __name__ == '__main__':
    unittest.main()
