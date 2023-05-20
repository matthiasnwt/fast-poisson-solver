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
import pickle
import random
import time
import pkg_resources
import numpy as np
import torch
import yaml
from matplotlib import rcParams
from importlib.resources import open_binary
from .model import PINN
from .utils import calculate_laplace, format_input

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['text.usetex'] = True


class Solver:
    """
    This class represents the main solver used for fast Poisson equation solving.
    It is a key component of the `fast_poisson_solver` package.

    Parameters
    ----------
    device : str, optional
        Specifies the device where the computations will be performed.
        This should be a valid PyTorch device string such as 'cuda' for GPU processing or 'cpu' for CPU processing.
        Default is 'cuda'.
    precision : torch.dtype, optional
        This determines the precision of the computation.
        This should be a valid PyTorch data type such as torch.float32 or torch.float64.
        Default is torch.float32.
    verbose : bool, optional
        When set to True, enables the printing of detailed logs during computation.
        Default is False.
    use_weights : bool, optional
        Determines whether the network uses pre-trained weights or random weights.
        If True, the network uses pre-trained weights. Default is True.
    compile_model : bool, optional
        Specifies whether the network model is compiled for faster inference.
        If False, the model won't be compiled. Default is True.
    lambdas_pde : list of float, optional
        A list that weights the influence of the PDE part in the loss term.
        If None, default weight 1e-12 will be used. Default is None.
    seed : int, optional
        This parameter sets the seed for generating random numbers,
        which helps in achieving deterministic results. Default is 0.
    """

    def __init__(self, device='cuda', precision=torch.float32, verbose=False,
                 use_weights=True, compile_model=True, lambdas_pde=None, seed=0):

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if lambdas_pde is None:
            lambdas_pde = [2 ** -12]
        self.path = os.path.join('resources', 'final.pt')

        self.verbose = verbose
        self.precision = precision
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_weights = use_weights
        self.compile_model = compile_model
        self.lambdas_pde = lambdas_pde
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.n_lambdas = len(self.lambdas_pde)

        # The losses for all the different values of lambda are put in those arrays
        self.Ls = np.zeros(self.n_lambdas)
        self.Ls_pde = np.zeros(self.n_lambdas)
        self.Ls_bc = np.zeros(self.n_lambdas)

        if self.path.endswith('.pt'):
            self.weights_input = True
            self.weights_path = pkg_resources.resource_stream(__name__, self.path)
            self.path = os.path.join(*os.path.split(self.path)[:1])
        else:
            self.weights_input = False

        # Path to the where the pre-computed data_utils will be saved.
        self.precompute_path = 'precomputed-resources'
        if not os.path.isdir(self.precompute_path):
            os.makedirs(self.precompute_path)
        self.precompute_file = os.path.join(self.precompute_path, 'default.pkl')

        self.load_data()
        self.build_model()

    def load_data(self):

        def tuple_constructor(loader, node):
            return tuple(loader.construct_sequence(node))

        yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

        path = pkg_resources.resource_stream(__name__, os.path.join(self.path, 'infos.yaml'))
        self.infos = yaml.safe_load(path)

        if 'out' not in self.infos['model']:
            self.infos['model']['out'] = self.infos['data_utils']['num_cases']

        if 'width' not in self.infos['model']:
            if 'network_width' in self.infos['model']:
                self.infos['model']['width'] = self.infos['model']['network_width']

        if 'depth' not in self.infos['model']:
            if 'network_depth' in self.infos['model']:
                self.infos['model']['depth'] = self.infos['model']['network_depth'] - 1

        # self.infos['model']['width'] = 800
        # self.infos['model']['depth'] = 8

    def build_model(self):

        self.model = PINN(self.infos['model']).to(self.device)

        if self.use_weights:
            state_dict = torch.load(self.weights_path, map_location=self.device)
            is_data_parallel = any('module' in key for key in state_dict.keys())
            is_compiled = any('_orig_mod' in key for key in state_dict.keys())
            if is_data_parallel:
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            if is_compiled:
                state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict)

        self.model.to(self.precision)

        if self.compile_model:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(e)
                print('Compiling did not work but do not worry, it will work without it.')
                print('You need pytorch 2.0 for compiling and Linux.')


    def evaluate_network_pde(self):
        self.x_pde.requires_grad = True
        self.y_pde.requires_grad = True
        self.H = self.model.h(self.x_pde, self.y_pde)
        self.DH = calculate_laplace(self.H, self.x_pde, self.y_pde).detach().to(self.precision)
        self.DHtDH = torch.matmul(self.DH.t(), self.DH)
        self.Dht = self.DH.t()
        self.x_pde.requires_grad = False
        self.y_pde.requires_grad = False
        self.H = self.H.detach()

    def evaluate_network_bc(self):
        with torch.no_grad():
            self.H_bc = self.model.h(self.x_bc, self.y_bc).to(self.precision)
        self.Ht_bc = self.H_bc.t()
        ones_Hbc = torch.ones(self.H_bc.shape[0], 1, device=self.device).to(self.precision)
        self.Ht_bc_ones = (self.Ht_bc @ ones_Hbc).t()

    def load_precomputed_data(self):
        with open(self.precompute_file, 'rb') as f:
            self.LHSs, self.RHSs, self.H, self.DH, self.H_bc, self.Ht_bc_ones, self.NdO, self.NO = format_input(
                pickle.load(f), self.precision, self.device, reshape=False)
        if self.verbose > 0:
            print('Pre-Computed data_utils loaded from storage.')

    def precompute_LHS_RHS(self):
        self.NO = self.DH.shape[0]
        self.NdO = self.H_bc.shape[0]
        M = torch.eye(self.NdO, device=self.device, dtype=self.precision) - \
            torch.ones(self.NdO, device=self.device, dtype=self.precision) / self.NdO

        lambdas = torch.tensor(self.lambdas_pde, device=self.device, dtype=self.precision).view(-1, 1, 1)

        self.LHSs = lambdas / self.NO * self.DHtDH + \
                    1 / self.NdO * torch.matmul(self.Ht_bc, torch.matmul(M, self.H_bc))
        self.RHSs = lambdas / self.NO * self.Dht
        if self.verbose > 0:
            print('Pre-Computed data_utils calculated.')

    def save_precomputed_data(self):
        with open(self.precompute_file, 'wb') as file:
            pickle.dump(
                [self.LHSs, self.RHSs, self.H, self.DH, self.H_bc, self.Ht_bc_ones, self.NdO, self.NO],
                file)
        if self.verbose > 0:
            print('Pre-Computed stored.')

    def precompute(self, x_pde, y_pde, x_bc, y_bc, name=None, save=False, load=False):
        """
        This method is used for precomputing of the data based on the given coordinates.

        Parameters
        ----------
        x_pde : tensor/array/list
            Coordinates that lie inside the domain and define the behavior of the partial differential equation (PDE).
        y_pde : tensor/array/list
            Coordinates that lie inside the domain and define the behavior of the partial differential equation (PDE).
        x_bc : tensor/array/list
            Coordinates of the boundary condition.
        y_bc : tensor/array/list
            Coordinates of the boundary condition.
        name : str, optional
            Name used for saving or loading the precomputed data.
            If no name is provided, the default name will be used. Default is None.
        save : bool, optional
            Specifies whether the precomputed data should be saved.
            If True, the data will be saved using the provided `name`. Default is True.
        load : bool, optional
            Specifies whether the precomputed data with the provided `name` should be loaded.
            If True, the method will attempt to load the data with the given name. Default is True.
        """
        t0 = time.perf_counter()

        if name is not None:
            self.precompute_file = os.path.join(self.precompute_path, name + f'.pkl')

        self.x_pde, self.y_pde, self.x_bc, self.y_bc = format_input([x_pde, y_pde, x_bc, y_bc],
                                                                    self.precision, self.device)

        self.x_tot = torch.cat([self.x_pde, self.x_bc], dim=0)
        self.y_tot = torch.cat([self.y_pde, self.y_bc], dim=0)

        assert torch.min(self.x_tot) >= 0 and torch.max(
            self.x_tot) <= 1, 'x coordinates should be in [0, 1]. Please rescale.'
        assert torch.min(self.y_tot) >= 0 and torch.max(
            self.y_tot) <= 1, 'y coordinates should be in [0, 1]. Please rescale.'

        if load and os.path.isfile(self.precompute_file):
            self.load_precomputed_data()
        else:
            self.evaluate_network_pde()
            self.evaluate_network_bc()
            self.precompute_LHS_RHS()
            if save:
                self.save_precomputed_data()

        # First time running torch.linalg.solve() is very slow, so we run it once here to get rid of the delay
        torch.linalg.solve(
            torch.rand(self.LHSs[0].shape).to(self.device).to(self.precision),
            torch.rand(self.LHSs[0].shape[0], 1).to(self.device).to(self.precision))

        t3 = time.perf_counter()
        if self.verbose > 0:
            print('\nPre-Computing Successful:', t3 - t0, 'seconds')

    def solve(self, f, bc):
        """
        This method is used to solve the PDE with the provided source function and boundary condition.

        Parameters
        ----------
        f : tensor/array/list
            The source function for the PDE.
        bc : tensor/array/list
            The boundary condition for the PDE.

        Returns
        -------
        tuple
            The tuple contains the following elements:

            u_pred : tensor
                The complete solution of the PDE.
            u_pde_pred : tensor
                The solution of the PDE inside the domain.
            u_bc_pred : tensor
                The solution for the boundary condition.
            f_pred : tensor
                The predicted source function.
            runtime : float
                The time it took the method to run in seconds.
        """
        self.f, self.bc = format_input([f, bc], self.precision, self.device)

        w_outs = []
        biases = np.empty(len(self.lambdas_pde))

        t0 = time.perf_counter()

        RHSs = torch.matmul(self.RHSs, self.f)
        self.bct_ones = torch.sum(self.bc).reshape(1, 1)

        for i, l in enumerate(self.lambdas_pde):
            self.w_out = torch.linalg.solve(self.LHSs[i], RHSs[i])
            self.bias = - 1 / self.NdO * (self.Ht_bc_ones @ self.w_out - self.bct_ones)

            # Just calculates the loss if multiple lambdas are provided to find the best one.
            if self.n_lambdas > 1:
                u_pred_bc = torch.matmul(self.H_bc, self.w_out) + self.bias
                f_pred = torch.matmul(self.DH, self.w_out)
                self.Ls_pde[i] = torch.mean((f_pred - self.f) ** 2).item()
                self.Ls_bc[i] = torch.mean((u_pred_bc - self.bc) ** 2).item()
                self.Ls[i] = self.Ls_pde[i] + self.Ls_bc[i]

                w_outs.append(self.w_out)
                biases[i] = self.bias

        if self.n_lambdas > 1:
            minimum = np.argmin(self.Ls)
            self.lambda_pde = self.lambdas_pde[minimum]
            self.w_out = w_outs[minimum]
            self.bias = biases[minimum]
        else:
            self.lambda_pde = self.lambdas_pde[0]

        self.u_pde_pred = torch.add(torch.matmul(self.H, self.w_out), self.bias)

        self.f_pred = torch.matmul(self.DH, self.w_out)
        self.u_bc_pred = torch.matmul(self.H_bc, self.w_out) + self.bias
        self.u_pred = torch.cat([self.u_pde_pred, self.u_bc_pred])

        t1 = time.perf_counter()
        if self.verbose > 0:
            print('\nRun Successful:', t1 - t0, 'seconds\n')

        return self.u_pred, self.u_pde_pred, self.u_bc_pred, self.f_pred, t1 - t0
