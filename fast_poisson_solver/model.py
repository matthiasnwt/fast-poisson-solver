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
from torch import nn

torch.manual_seed(1)


class PINN(nn.Module):
    def __init__(self, data):
        super(PINN, self).__init__()

        self.width = data['width']
        self.depth = data['depth']
        self.ffm_params = data['ffm']
        self.activation_fn = data['activation_fn']
        self.dropout = data['dropout']
        self.layer_out = data['out']
        self.use_bn = data['use_bn']
        self.weights_init = data['weights_init']

        if self.ffm_params is None:
            self.ffm = False
            self.layers_h = [2] + [self.width] * self.depth
            self.depth += 1
        else:
            self.ffm = True
            self.layers_h = [self.width] * self.depth
            self.ffm_params['num_inputs'] = 2
            self.ffm_params['num_outputs'] = self.layers_h[0]
            self.ffm_params['activation_fn'] = self.activation_fn
            self.mapping = FourierFeatureMapping(self.ffm_params)

        # Build model
        self.model = []
        for i in range(self.depth - 1):
            if i == 0 and self.ffm:
                self.model.append(nn.Identity())
            self.model.append(nn.Linear(self.layers_h[i], self.layers_h[i + 1]))

            self.model.append(nn.Dropout(self.dropout))

            if self.use_bn:
                self.model.append(nn.BatchNorm1d(self.layers_h[i + 1]))

            self.model.append(get_activation(self.activation_fn, self.layers_h[i + 1]))

        # Make model sequential
        self.model = nn.ModuleList(self.model)
        self.model = nn.Sequential(*self.model)

        # Output Layer
        self.out = nn.Linear(self.layers_h[-1], self.layer_out, bias=True)

    def forward(self, x, y):
        xy = self.h(x, y)
        xy = self.out(xy)
        return xy

    def h(self, x, y):
        xy = torch.cat((x, y), dim=1)
        if self.ffm:
            xy = self.mapping(xy)
        xy = self.model(xy)
        return xy

    def get_infos(self):
        infos = {
            'network': self.layers_h,
            'width': self.width,
            'depth': self.depth,
            'activation_fn': self.activation_fn,
            'dropout': self.dropout,
            'use_bn': self.use_bn,
            'weights_init': self.weights_init,
            'ffm': self.ffm_params,
            'out': self.layer_out,
        }
        return infos


class Stan(torch.nn.Module):
    def __init__(self, size, use_scaling=True, initial_beta=1.0, initial_scale=1.0):
        super().__init__()
        if use_scaling:
            self.scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.scale = 1.0

        self.beta = nn.Parameter(torch.tensor([initial_beta] * size))
        self.beta_history = []

    def forward(self, x):
        self.beta_history.append(self.beta.clone().detach())
        x = torch.tanh(self.scale * x) + self.beta * x * torch.tanh(self.scale * x)
        return x


class FourierFeatureMapping(nn.Module):
    def __init__(self, ffm_params):
        super(FourierFeatureMapping, self).__init__()

        self.num_inputs = ffm_params['num_inputs']
        self.num_outputs = ffm_params['num_outputs']
        self.num_features = ffm_params['num']
        self.activation_fn = ffm_params['activation_fn']
        self.scale = float(ffm_params['scale'])
        self.requires_grad = ffm_params['req_grad']

        self.freqs = nn.Parameter(torch.randn(self.num_inputs, self.num_features) / self.num_inputs * self.scale,
                                  requires_grad=self.requires_grad)

        self.coefficients = nn.Parameter(torch.ones(self.num_features), requires_grad=self.requires_grad)
        self.biases = nn.Parameter(torch.zeros(self.num_features), requires_grad=self.requires_grad)

        self.linear = nn.Linear(self.num_features * 2, self.num_outputs)

        self.activation = get_activation(self.activation_fn, self.num_outputs)

    def forward(self, x):
        # Compute Fourier features
        fourier_sin = self.coefficients * torch.sin(torch.matmul(x, self.freqs) + self.biases)
        fourier_cos = self.coefficients * torch.cos(torch.matmul(x, self.freqs) + self.biases)

        fourier = torch.cat((fourier_sin, fourier_cos), dim=1)

        fourier = fourier.reshape(fourier.shape[0], -1)

        # Pass through linear layer
        out = self.activation(self.linear(fourier))

        return out


def get_activation(n, size=None):
    if n == 'stan':
        return Stan(size)
    elif n == 'stan1':
        return Stan(size, False)
    elif n == 'tanh':
        return nn.Tanh()
    elif n == 'sigmoid':
        return nn.Sigmoid()
    elif n == 'softplus':
        return nn.Softplus()
    elif n == 'elu':
        return nn.ELU()
    elif n == 'silu':
        return nn.SiLU()
    elif n == 'gelu':
        return nn.GELU()
    elif n == 'softsign':
        return nn.Softsign()
    elif n == 'logsigmoid':
        return nn.LogSigmoid()
    elif n == 'tanhshrink':
        return nn.Tanhshrink()
    elif n == 'mish':
        return nn.Mish()
    else:
        print(f'Activation Function {n} not defined. Using Tanh instead.')
        return nn.Tanh()


if __name__ == '__main__':
    device = 'cuda'
    precision = torch.float16
    model_data = {
        'use_bn': False,
        'activation_fn': 'tanh',
        'ffm': {
            'req_grad': False,
            'num': 200,
            'scale': 10,
        },

        'dropout': 0,
        'weights_init': None,

        'width': 700,
        'depth': 6,
        'out': 800
    }

    model = PINN(model_data).to(device).to(precision)

    # analyze_ntk(model)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number trainable Parameters:", total_params)

    total_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number non-trainable Parameters:", total_params)

    for p in model.parameters():
        if not p.requires_grad:
            print(p.size())
