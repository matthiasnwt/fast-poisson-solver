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
import time

import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score

from .utils import format_input



def analyze(f, f_pred, u_pred, u_bc_pred, u_num, u_bc, normalize=True):
    t0 = time.perf_counter()

    f, f_pred, u_pred, u_bc_pred, u_num, u_bc = format_input([f, f_pred, u_pred, u_bc_pred, u_num, u_bc], as_array=True)

    f, f_pred, u_pred, u_bc_pred, u_num, u_bc = [v.reshape(-1) for v in [f, f_pred, u_pred, u_bc_pred, u_num, u_bc]]

    if normalize:
        bv = u_bc[0]
    else:
        bv = 0

    def analyze_struct(image1, image2, bv):
        image1_ = image1 - bv
        image2_ = image2 - bv
        maximum = np.max(image1_)
        minimum = np.min(image1_)

        if len(np.unique(image1_)) != 1 and normalize:
            image1_ = (image1_ - minimum) / (maximum - minimum)
            image2_ = (image2_ - minimum) / (maximum - minimum)
            range = 1
        else:
            range = np.min((maximum - minimum, 1e-8))

        ssim_value = ssim(image1_, image2_, data_range=range+1e-8, multichannel=False, gaussian_weights=True)

        psnr_value = psnr(image1_, image2_, data_range=range)
        r2 = r2_score(image1_.reshape(-1), image2_.reshape(-1))
        return ssim_value, psnr_value, r2

    def analyze_standard(image1, image2, bv):
        image1_ = image1 - bv
        image2_ = image2 - bv
        maximum = np.max(image1_)
        minimum = np.min(image1_)
        if len(np.unique(image1_)) != 1 and normalize:
            image1_ = (image1_ - minimum) / (maximum - minimum)
            image2_ = (image2_ - minimum) / (maximum - minimum)
        mse = np.mean((image1_ - image2_) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(image1_ - image2_))
        mae_r = np.mean(np.abs(image1_ - image2_)) / np.max((np.mean(np.abs(image1_)), 1e-6))
        return mse, rmse, mae, mae_r

    ssim_value_u, psnr_value_u, r2_u = analyze_struct(u_num, u_pred, bv)
    mse_u, rmse_u, mae_u, mae_r_u = analyze_standard(u_num, u_pred, bv)

    ssim_value_f, psnr_valuie_f, r2_f = analyze_struct(f, f_pred, 0)
    mse_f, rmse_f, mae_f, mae_r_f = analyze_standard(f, f_pred, 0)

    mse_u_bc, rmse_u_bc, mae_u_bc, mae_r_bc = analyze_standard(u_bc, u_bc_pred, bv)

    res = {
        'u': {
            'MSE': mse_u,
            'RMSE': rmse_u,
            'MAE': mae_u,
            'MAE_R': mae_r_u,
            'SSIM': ssim_value_u,
            'PSNR': psnr_value_u,
            'R2': r2_u
        },
        'f': {
            'MSE': mse_f,
            'RMSE': rmse_f,
            'MAE': mae_f,
            'MAE_R': mae_r_f,
            'SSIM': ssim_value_f,
            'PSNR': psnr_valuie_f,
            'R2': r2_f
        },
        'bc': {
            'MSE': mse_u_bc,
            'RMSE': rmse_u_bc,
            'MAE': mae_u_bc,
            'MAE_R': mae_r_bc
        },
    }

    return res
