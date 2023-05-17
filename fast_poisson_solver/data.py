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


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc

from .data_utils import source_functions as sf


class Data:
    def __init__(self, cases, domain_x=None, domain_y=None, grid_num=32, noise_std=0, shuffle=False,
                 initial_shuffle=False, batchsize=-1, batchsize_bc=-1,
                 use_torch=False, device='cpu', precision=torch.float32, random_coords=False, seed=0):
        if domain_y is None:
            domain_y = [0, 1]
        if domain_x is None:
            domain_x = [0, 1]
        self.y = None
        self.x = None
        self.y_grid = None
        self.x_grid = None
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.grid_num = grid_num
        self.cases = cases
        self.noise_std = noise_std
        self.num_cases = len(cases)
        self.shuffle = shuffle
        self.initial_shuffle = initial_shuffle
        self.batchsize = batchsize if batchsize > 0 else self.grid_num ** 2
        self.batchsize_bc = batchsize_bc if batchsize_bc > 0 else 4 * self.grid_num + 4
        self.batch_number = 0
        self.number_of_batches = np.ceil(self.grid_num ** 2 / self.batchsize).astype(int)
        self.use_torch = use_torch
        self.device = device
        self.random_coords = random_coords
        self.seed = seed
        self.precision = precision

        assert 0 <= self.domain_y[0] <= 1 and 0 <= self.domain_y[1] <= 1, \
            "Both elements of domain_y should lie within the interval [0, 1] inclusive"
        assert self.domain_y[1] > self.domain_y[0], \
            "The second element of domain_y should be larger than the first element"
        assert 0 <= self.domain_x[0] <= 1 and 0 <= self.domain_x[1] <= 1, \
            "Both elements of domain_x should lie within the interval [0, 1] inclusive"
        assert self.domain_x[1] > self.domain_x[0], \
            "The second element of domain_y should be larger than the first element"

        x_domain_length = self.domain_x[1] - self.domain_x[0]
        y_domain_length = self.domain_y[1] - self.domain_y[0]
        if x_domain_length > y_domain_length:
            self.x_grid_num = self.grid_num
            self.y_grid_num = int(self.grid_num * y_domain_length / x_domain_length)
        elif x_domain_length < y_domain_length:
            self.x_grid_num = int(self.grid_num * x_domain_length / y_domain_length)
            self.y_grid_num = self.grid_num
        else:
            self.x_grid_num = self.grid_num
            self.y_grid_num = self.grid_num

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # self.indices = [np.random.choice(self.grid_num**2, self.batchsize, replace=False) for _ in range(self.num_cases)]
        # self.indices = np.arange(self.grid_num ** 2)
        self.indices = np.arange(self.x_grid_num * self.y_grid_num)
        # self.indices_bc = np.arange(4 * self.grid_num + 4)
        self.indices_bc = np.arange(2 * self.x_grid_num + 2 * self.y_grid_num + 4)

        if self.initial_shuffle:
            self.indices = np.random.permutation(self.indices)
            self.indices_bc = np.random.permutation(self.indices_bc)

        self.generate_boundary_coords()

        if not random_coords:
            self.generate_grid()
        else:
            self.generate_random_coords()
        self.calculate_cases()

        if self.use_torch:
            self.to_torch_tensors()
        else:
            self.x_grid = self.x_grid.reshape(-1, 1)
            self.y_grid = self.x_grid.reshape(-1, 1)
            self.x_bc = self.x_grid.reshape(-1, 1)
            self.y_bc = self.x_grid.reshape(-1, 1)

    def get_infos(self):
        data = {
            'domain_x': self.domain_x,
            'domain_y': self.domain_y,
            'grid_num': self.grid_num,
            'num_cases': self.num_cases,
            'shuffle': self.shuffle,
            'initial_shuffle': self.initial_shuffle,
            'batchsize': self.batchsize,
            'batchsize_bc': self.batchsize_bc,
            'use_torch': self.use_torch,
            'device': self.device,
            'seed': self.seed,
            'precision': str(self.precision),
            'cases': self.cases,
        }
        return data

    def to_torch_tensors(self):
        self.out = torch.from_numpy(self.out).to(self.device).to(self.precision)
        self.x_grid = torch.from_numpy(self.x_grid).unsqueeze(1).to(self.device).to(self.precision)
        self.y_grid = torch.from_numpy(self.y_grid).unsqueeze(1).to(self.device).to(self.precision)
        self.bc = torch.from_numpy(self.bc).to(self.device).to(self.precision)
        self.x_bc = torch.from_numpy(self.x_bc).unsqueeze(1).to(self.device).to(self.precision)
        self.y_bc = torch.from_numpy(self.y_bc).unsqueeze(1).to(self.device).to(self.precision)
        self.indices = torch.from_numpy(self.indices).long().to(self.device)
        self.indices_bc = torch.from_numpy(self.indices_bc).long().to(self.device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_number < self.number_of_batches:
            out_call, x_call, y_call, bc_call, x_bc_call, y_bc_call = self.__call__(self.batch_number)
            self.batch_number += 1
            return out_call, x_call, y_call, bc_call, x_bc_call, y_bc_call
        else:
            self.batch_number = 0
            raise StopIteration

    def __call__(self, batch_number):
        if batch_number == 0 and self.shuffle:
            self.shuffle_epoch()

        min_batch = batch_number * self.batchsize
        max_batch = min((min_batch + self.batchsize, self.grid_num ** 2))
        max_batch = min((min_batch + self.batchsize, self.x_grid_num * self.y_grid_num))
        batchsize_i = max_batch - min_batch

        min_batch_bc = batch_number * self.batchsize_bc
        # max_batch_bc = min((min_batch_bc + self.batchsize_bc, 4 * self.grid_num + 4))
        max_batch_bc = min((min_batch_bc + self.batchsize_bc, 2 * self.x_grid_num + 2 * self.y_grid_num + 4))
        batchsize_i_bc = max_batch_bc - min_batch_bc

        indices_call = self.indices[min_batch: max_batch]
        if self.use_torch:
            out_call = torch.zeros(size=(batchsize_i, self.num_cases), device=self.device, dtype=self.precision)
        else:
            out_call = np.zeros((batchsize_i, self.num_cases))
        x_call = self.x_grid[indices_call]
        y_call = self.y_grid[indices_call]
        for i, out_i in enumerate(self.out.T):
            out_call[:, i] = out_i[indices_call]

        indices_call_bc = self.indices_bc[min_batch_bc: max_batch_bc]
        if self.use_torch:
            bc_call = torch.zeros(size=(batchsize_i_bc, self.num_cases), device=self.device, dtype=self.precision)
        else:
            bc_call = np.zeros((batchsize_i_bc, self.num_cases))

        x_bc_call = self.x_bc[indices_call_bc]
        y_bc_call = self.y_bc[indices_call_bc]
        for i, bc_i in enumerate(self.bc.T):
            bc_call[:, i] = bc_i[indices_call_bc]

        return out_call, x_call, y_call, bc_call, x_bc_call, y_bc_call

    def shuffle_epoch(self):
        if self.use_torch:
            self.indices = torch.randperm(self.indices.shape[0])
            self.indices_bc = torch.randperm(self.indices_bc.shape[0])
        else:
            self.indices = np.random.permutation(self.indices)
            self.indices_bc = np.random.permutation(self.indices_bc)

    def generate_grid(self):
        # x = np.linspace(self.domain_x[0], self.domain_x[1], self.grid_num + 2)[1:-1]
        x = np.linspace(self.domain_x[0], self.domain_x[1], self.x_grid_num + 2)[1:-1]
        y = np.linspace(self.domain_y[0], self.domain_y[1], self.y_grid_num + 2)[1:-1]
        x, y = np.meshgrid(x, y)
        self.x_grid = x.flatten()
        self.y_grid = y.flatten()

    def generate_random_coords(self):
        engine = qmc.Sobol(d=2, scramble=True, seed=self.seed)  # d=2 for 2D points
        sample = engine.random(int(self.grid_num ** 2))
        self.x_grid, self.y_grid = np.hsplit(sample, 2)
        self.y_grid = self.y_grid.flatten()
        self.x_grid = self.x_grid.flatten()

    def generate_boundary_coords(self):
        # x_grid = np.linspace(self.domain_x[0], self.domain_x[1], self.grid_num + 2)
        x_grid = np.linspace(self.domain_x[0], self.domain_x[1], self.x_grid_num + 2)
        y_grid = np.linspace(self.domain_y[0], self.domain_y[1], self.y_grid_num + 2)[1:-1]
        lower_x = np.ones_like(y_grid) * self.domain_x[0]
        lower_y = np.ones_like(x_grid) * self.domain_y[0]
        upper_x = np.ones_like(y_grid) * self.domain_x[1]
        upper_y = np.ones_like(x_grid) * self.domain_y[1]

        self.x_bc = np.concatenate([x_grid, x_grid, lower_x, upper_x]).flatten()
        self.y_bc = np.concatenate([lower_y, upper_y, y_grid, y_grid]).flatten()

    def calculate_cases(self):
        self.out = np.zeros((len(self.x_grid), len(self.cases)))
        self.bc = np.zeros((len(self.x_bc), len(self.cases)))

        for i, case in enumerate(self.cases):
            if case['b_val'] == 'random':
                self.bc[:, i] = np.ones_like(self.x_bc) * np.random.uniform(-10, 10)
            else:
                np.random.uniform(-10, 10)  # to make sure the random seed is not affected
                self.bc[:, i] = np.ones_like(self.x_bc) * case['b_val']

            # if case['name'] == 'sinsin':
            #     self.out[:, i] = sf.sinsin(self.x_grid, self.y_grid, case['param'])
            if case['name'] == 'sin':
                self.out[:, i] = sf.sincos(self.x_grid, self.y_grid, case['param'])
            elif case['name'] == 'exp':
                self.out[:, i] = sf.exp(self.x_grid, self.y_grid, case['param'])
            elif case['name'] == 'perlin':
                self.out[:, i] = sf.perlin(self.x_grid, self.y_grid, case['param'])
            # elif case['name'] == 'rectangle':
            #     self.out[:, i] = sf.rectangle(self.x_grid, self.y_grid, case['param'])
            # elif case['name'] == 'circle':
            #     self.out[:, i] = sf.circle(self.x_grid, self.y_grid, case['param'])
            elif case['name'] == 'geo':
                self.out[:, i] = sf.rectangle_circle(self.x_grid, self.y_grid, case['param'], grid_num=self.grid_num)
            else:
                raise ValueError('Unknown case')

            if self.noise_std > 0:
                self.out[:, i] *= np.random.normal(1, self.noise_std, len(self.x_grid))

    def plot_functions(self):

        f, x, y, *_ = self.__call__(0)

        if self.use_torch:
            f = f.float().cpu().detach().numpy()
            x = x.float().cpu().detach().numpy()
            y = y.float().cpu().detach().numpy()

        xy = np.array([x, y]).T
        ind = np.lexsort((xy[:, 1], xy[:, 0]))

        f = f[ind]
        x = x[ind]
        y = y[ind]

        def plot(fig, ax, x, y, v, title):
            x = x.reshape(self.grid_num, self.grid_num)
            y = y.reshape(self.grid_num, self.grid_num)
            v = v.reshape(self.grid_num, self.grid_num)
            c = ax.contourf(x, y, v, 100, cmap='jet')
            # c = ax.scatter(x, y, c=v, s=1)
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(c, ax=ax)

        plots_x = np.min((self.num_cases, 4))
        plots_y = int(np.ceil(self.num_cases / plots_x))

        fig, axs = plt.subplots(plots_y, plots_x, figsize=(plots_x * 4, plots_y * 4), dpi=100, tight_layout=True)

        if self.num_cases == 1:
            plot(fig, axs, x, y, f[:, 0], self.cases[0]['name'])
        else:
            i = 0
            for ax in axs.flatten():
                if i < self.num_cases:
                    plot(fig, ax, x, y, f[:, i], self.cases[i]['name'])
                else:
                    ax.axis('off')
                i += 1

        plt.savefig(os.path.join('..', 'images', 'functions.png'))
        plt.close()


if __name__ == '__main__':
    # cases = [{'name': 'sinsin', 'param': ([1, 1], [0, 0], 1)},
    #          {'name': 'sinsin', 'param': ([1, 3], [0, 0], 1)},
    #          {'name': 'sinsin', 'param': ([3, 1], [0, 0], 1)},
    #          {'name': 'sinsin', 'param': ([4, 4], [0, 0], 1)}
    #          ]
    # {'name': 'sinsin', 'param': ([5, 6], 1)},
    # {'name': 'sinsin', 'param': ([6, 5], 1)},
    # {'name': 'sincos', 'param': ([1, 1], 1)},
    # {'name': 'sincos', 'param': ([1, 3], 1)},
    # {'name': 'sincos', 'param': ([3, 1], 1)},
    # {'name': 'sincos', 'param': ([4, 4], 1)},
    # {'name': 'sincos', 'param': ([5, 6], 1)},
    # {'name': 'sincos', 'param': ([6, 5], 1)},
    # {'name': 'exp', 'param': 2}

    cases = [{'name': 'perlin', 'param': 'random', 'b_val': 100}]
    # cases = [{'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          {'name': 'perlin', 'param': 'random'},
    #          ]

    # cases = [{'name': 'sinsin', 'param': 'random'},
    #          {'name': 'sinsin', 'param': 'random'},
    #          {'name': 'sinsin', 'param': 'random'},
    #          {'name': 'sinsin', 'param': 'random'},
    #          {'name': 'sinsin', 'param': 'random'},
    #          {'name': 'sincos', 'param': 'random'},
    #          {'name': 'sincos', 'param': 'random'},
    #          {'name': 'sincos', 'param': 'random'},
    #          {'name': 'sincos', 'param': 'random'},
    #          {'name': 'sincos', 'param': 'random'},
    #          {'name': 'exp', 'param': 'random'}]

    data = Data(domain_x=[0, 1],
                domain_y=[0, 1],
                grid_num=50,
                cases=cases,
                shuffle=False,
                add_noise=False,
                initial_shuffle=False,
                batchsize=-1,
                batchsize_bc=-1,
                use_torch=False,
                device='cuda',
                precision=torch.float32,
                seed=2)

    roh_pde, x_pde, y_pde, u_bc, x_bc, y_bc = data.__call__(0)

    print(roh_pde.shape, x_pde.shape, y_pde.shape, u_bc.shape, x_bc.shape, y_bc.shape)

    # data_utils.plot_functions()

    # for i, (out, x, y) in enumerate(data_utils):
    #     print(i)
    # out, x, y = data_utils.__next__()
    # print(out)
    # plt.scatter(x[:, 0], y[:, 0], c=out[:, 0], s=2, cmap='jet')
    # plt.savefig(f'temp.png')
    # plt.close()
    #
    # for i, (out, x, y) in enumerate(data_utils):
    #     print(i)
    #     plt.scatter(x[:, 0], y[:, 0], c=out[:, 0], s=2, cmap='jet')
    #     plt.savefig(f'temp{i}_.png')
    #     plt.close()

    # data_utils()
