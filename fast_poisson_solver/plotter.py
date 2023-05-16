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

import matplotlib.font_manager as fm
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from .utils import minmax, format_input, process_grid_data


#
# def plot_lambda_error(solver):
#     minimum = np.argmin(solver.Ls)
#     solver.lambda_pde = solver.lambdas_pde[minimum]
#     minimum = solver.Ls[minimum]
#
#     minimum_pde = np.min(solver.Ls_pde)
#     minimum_bc = np.min(solver.Ls_bc)
#
#     def plot_lambdas(ax, d, ylim, ylabel=False):
#         ax.plot(solver.lambdas_pde, d)
#         ax.set_ylim(ylim)
#     # ax.axvline(100, color='red', label='$\lambda=100$')
#     ax.set_xlabel('$\lambda$', fontsize=18)
#     if ylabel: ax.set_ylabel('Loss', fontsize=18)
#     # ax.legend(fontsize=18, loc='lower right')
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300, figsize=(15, 5), tight_layout=True)
#
# plot_lambdas(ax1, solver.Ls, (0, minimum * 2), ylabel=True)
# plot_lambdas(ax2, solver.Ls_pde, (0, minimum_pde * 2))
# plot_lambdas(ax3, solver.Ls_bc, (0, minimum_bc * 5))
#
# ax1.text(0.5, -0.25, '(a)', transform=ax1.transAxes, ha='center', va='center', fontsize=18)
# ax2.text(0.5, -0.25, '(b)', transform=ax2.transAxes, ha='center', va='center', fontsize=18)
# ax3.text(0.5, -0.25, '(c)', transform=ax3.transAxes, ha='center', va='center', fontsize=18)
#
# ax1.set_xscale('log', base=2)
# ax2.set_xscale('log', base=2)
# ax3.set_xscale('log', base=2)
# plt.savefig('../Auswertungen/Lambda_Transfer/Lambdas_Transfer.png')
# plt.savefig('../Auswertungen/Lambda_Transfer/Lambdas_Transfer.pdf')
#
# plt.show()


def plot_subplot(ax, x, y, v, title, vmin=None, vmax=None, cb_pad=0.018, cb_ztick=False, grid=False, show_points=False):
    if vmin is None:
        vmin = min((np.min(v), 0))
    if vmax is None:
        vmax = max((np.max(v), 0))

    if grid:
        c = ax.imshow(v, cmap='jet', vmin=vmin, vmax=vmax, extent=(0, 1, 0, 1), origin='lower')
    else:
        c = ax.tricontourf(x.reshape(-1), y.reshape(-1), v.reshape(-1), 100, cmap='jet', vmin=vmin,
                           vmax=vmax)  # , extent=(0, 1, 0, 1))

    if show_points:
        ax.scatter(x, y, s=1, c='black', marker='.', alpha=0.5)

    if title != '':
        ax.set_title(title, fontsize=16, pad=12, fontweight='bold')

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks((0, 1), labels=('0', '1'), fontsize=14)
    ax.set_yticks((0, 1), labels=('0', '1'), fontsize=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = plt.colorbar(ScalarMappable(norm=c.norm, cmap=c.cmap), ax=ax, extendrect=True, pad=cb_pad,
                      location='bottom', shrink=0.8)
    cb.outline.set_visible(False)

    if cb_ztick:
        cb.set_ticks([vmin, 0, vmax], labels=[f'{vmin:.4f}', '0', f'{vmax:.4f}'], fontsize=15)
    else:
        cb.set_ticks([vmin, vmax], labels=[f'{vmin:.4f}', f'{vmax:.4f}'], fontsize=15)


def plot_comparison(x_pde, y_pde, x_bc, y_bc, u_pred, f, f_pred, u_num,
                    grid=False, save=False, save_path=None, name=None, show=True):
    """
    This function is used to plot and compare the numeric solution, the predicted Machine Learning solution,
    and the residual between the two. It also shows the true source function, the predicted source function,
    and the residual between these two.

    Parameters
    ----------
    x_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    y_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    x_bc : tensor/array/list
        Coordinates of the boundary condition.
    y_bc : tensor/array/list
        Coordinates of the boundary condition.
    u_pred : tensor/array/list
        The predicted solution of the PDE using Machine Learning.
    f : tensor/array/list
        The true source function for the PDE.
    f_pred : tensor/array/list
        The predicted source function for the PDE.
    u_num : tensor/array/list
        The numeric solution of the PDE.
    grid : bool, optional
        If True, the data is arranged into a grid and plotted as an image.
        If False, tricontourf is used to create a contour plot. Default is False.
    save : bool, optional
        Whether to save the image. The image is saved in both .pdf and .png formats. Default is False.
    save_path : str, optional
        Path where the image will be saved. Used only if `save` is True. Default is None.
    name : str, optional
        Name of the image file. Used only if `save` is True. Default is None.
    show : bool, optional
        Whether to display the plot. Default is False.
    """

    u_pred, u_num, f, f_pred, x, y, x_bc, y_bc = format_input([u_pred, u_num, f, f_pred, x_pde, y_pde, x_bc, y_bc],
                                                              precision=torch.float64, device="cpu", as_array=True)

    x_tot = np.concatenate([x, x_bc])
    y_tot = np.concatenate([y, y_bc])

    if grid:
        x_tot, y_tot, u_num, u_pred = process_grid_data(x_tot, y_tot, u_num, u_pred)
        x, y, f, f_pred = process_grid_data(x, y, f, f_pred)

    vmin_u, vmax_u = minmax(u_pred, u_num)
    vmin_f, vmax_f = minmax(f_pred, f)

    fig, axs = plt.subplots(2, 3, figsize=(10, 8), dpi=400, tight_layout=True, sharey='row', sharex='col')

    axs[0][0].text(-0.15, 0.5, 'Potential', ha='center', va='center', rotation='vertical', fontsize=16,
                   fontweight='bold')
    axs[1][0].text(-0.15, 0.5, 'Source Function', ha='center', va='center', rotation='vertical', fontsize=16,
                   fontweight='bold')

    plot_subplot(axs[0][0], x_tot, y_tot, u_num, 'Numeric', vmin_u, vmax_u, cb_pad=0.03, grid=grid)
    plot_subplot(axs[0][1], x_tot, y_tot, u_pred, 'Machine Learning', vmin_u, vmax_u, cb_pad=0.03, grid=grid)
    plot_subplot(axs[0][2], x_tot, y_tot, (u_pred - u_num), 'Residual', cb_pad=0.03, cb_ztick=True, grid=grid)
    plot_subplot(axs[1][0], x, y, f, '', vmin_f, vmax_f, cb_pad=0.08, grid=grid)
    plot_subplot(axs[1][1], x, y, f_pred, '', vmin_f, vmax_f, cb_pad=0.08, grid=grid)
    plot_subplot(axs[1][2], x, y, (f_pred - f), '', cb_pad=0.08, cb_ztick=True, grid=grid)

    axs[0][0].set_ylabel('y', labelpad=-10, fontsize=14)
    axs[1][0].set_ylabel('y', labelpad=-10, fontsize=14)
    axs[1][0].set_xlabel('x', labelpad=-15, fontsize=14)
    axs[1][1].set_xlabel('x', labelpad=-15, fontsize=14)
    axs[1][2].set_xlabel('x', labelpad=-15, fontsize=14)

    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, name + '.pdf'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, name + '.png'), bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    # np.save(os.path.join(save_path, name + '_residual.npy'), (u_pred - u_num).reshape(solver.grid_size, solver.grid_size))


def plot(x_pde, y_pde, x_bc, y_bc, u_pred, f, f_pred, grid=False, save=False, save_path=None, name=None, show=True):
    """
    This function is used to plot the predicted Machine Learning solution of the PDE,
    the true source function, and the predicted source function.

    Parameters
    ----------
    x_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    y_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    x_bc : tensor/array/list
        Coordinates of the boundary condition.
    y_bc : tensor/array/list
        Coordinates of the boundary condition.
    u_pred : tensor/array/list
        The predicted solution of the PDE using Machine Learning.
    f : tensor/array/list
        The true source function for the PDE.
    f_pred : tensor/array/list
        The predicted source function for the PDE.
    grid : bool, optional
        If True, the data is arranged into a grid and plotted as an image.
        If False, tricontourf is used to create a contour plot. Default is False.
    save : bool, optional
        Whether to save the image. The image is saved in both .pdf and .png formats. Default is False.
    save_path : str, optional
        Path where the image will be saved. Used only if `save` is True. Default is None.
    name : str, optional
        Name of the image file. Used only if `save` is True. Default is None.
    show : bool, optional
        Whether to display the plot. Default is False.

    """
    u_pred, f, f_pred, x, y, x_bc, y_bc = format_input([u_pred, f, f_pred, x_pde, y_pde, x_bc, y_bc],
                                                       precision=torch.float64, device="cpu", as_array=True)

    x_tot = np.concatenate([x, x_bc])
    y_tot = np.concatenate([y, y_bc])

    if grid:
        x_tot, y_tot, u_pred = process_grid_data(x_tot, y_tot, u_pred)
        x, y, f, f_pred = process_grid_data(x, y, f, f_pred)

    vmin_u, vmax_u = minmax(u_pred)
    vmin_f, vmax_f = minmax(f_pred, f)

    fig, axs = plt.subplots(2, 3, figsize=(10, 8), dpi=400, tight_layout=True, sharey='row', sharex='col')

    axs[0][0].text(-0.15, 0.5, 'Potential', ha='center', va='center', rotation='vertical', fontsize=16,
                   fontweight='bold')
    axs[1][0].text(-0.15, 0.5, 'Source Function', ha='center', va='center', rotation='vertical', fontsize=16,
                   fontweight='bold')

    plot_subplot(axs[0][1], x_tot, y_tot, u_pred, 'Machine Learning', vmin_u, vmax_u, cb_pad=0.03, grid=grid)
    plot_subplot(axs[1][0], x, y, f, '', vmin_f, vmax_f, cb_pad=0.08, grid=grid, show_points=True)
    plot_subplot(axs[1][1], x, y, f_pred, '', vmin_f, vmax_f, cb_pad=0.08, grid=grid)
    plot_subplot(axs[1][2], x, y, (f_pred - f), '', cb_pad=0.08, cb_ztick=True, grid=grid)

    axs[0][0].set_ylabel('y', labelpad=-10, fontsize=14)
    axs[1][0].set_ylabel('y', labelpad=-10, fontsize=14)
    axs[1][0].set_xlabel('x', labelpad=-15, fontsize=14)
    axs[1][1].set_xlabel('x', labelpad=-15, fontsize=14)
    axs[1][2].set_xlabel('x', labelpad=-15, fontsize=14)

    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, name + '.pdf'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, name + '.png'), bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_side_by_side(x_pde, y_pde, x_bc, y_bc, u_pred, f, f_pred, u_num,
                      grid=False, save=False, save_path=None, name=None, show=True):
    """
    This function is used to plot and compare the numeric solution, the predicted Machine Learning solution,
    and the residual between the two. It also shows the true source function, the predicted source function,
    and the residual between these two.

    Parameters
    ----------
    x_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    y_pde : tensor/array/list
        Coordinates that lie inside the domain and define the behavior of the PDE.
    x_bc : tensor/array/list
        Coordinates of the boundary condition.
    y_bc : tensor/array/list
        Coordinates of the boundary condition.
    u_pred : tensor/array/list
        The predicted solution of the PDE using Machine Learning.
    f : tensor/array/list
        The true source function for the PDE.
    f_pred : tensor/array/list
        The predicted source function for the PDE.
    u_num : tensor/array/list
        The numeric solution of the PDE.
    grid : bool, optional
        If True, the data is arranged into a grid and plotted as an image.
        If False, tricontourf is used to create a contour plot. Default is False.
    save : bool, optional
        Whether to save the image. The image is saved in both .pdf and .png formats. Default is False.
    save_path : str, optional
        Path where the image will be saved. Used only if `save` is True. Default is None.
    name : str, optional
        Name of the image file. Used only if `save` is True. Default is None.
    show : bool, optional
        Whether to display the plot. Default is False.
    """

    u_pred, u_num, f, f_pred, x, y, x_bc, y_bc = format_input([u_pred, u_num, f, f_pred, x_pde, y_pde, x_bc, y_bc],
                                                              precision=torch.float64, device="cpu", as_array=True)

    x_tot = np.concatenate([x, x_bc])
    y_tot = np.concatenate([y, y_bc])

    if grid:
        x_tot, y_tot, u_num, u_pred = process_grid_data(x_tot, y_tot, u_num, u_pred)
        x, y, f, f_pred = process_grid_data(x, y, f, f_pred)

    vmin_u, vmax_u = minmax(u_pred, u_num)
    vmin_f, vmax_f = minmax(f_pred, f)

    fig, ax = plt.subplots(1, 2, figsize=(10, 8), dpi=400, tight_layout=True, sharey='row', sharex='col')

    ax[0].tricontourf(x_tot.reshape(-1), y_tot.reshape(-1), u_num.reshape(-1), 200, cmap='jet', vmin=vmin_u,
                      vmax=vmax_u)
    ax[1].tricontourf(x_tot.reshape(-1), y_tot.reshape(-1), u_pred.reshape(-1), 200, cmap='jet', vmin=vmin_u,
                      vmax=vmax_u)

    font_name = "Calibri" if "Calibri" in fm.findSystemFonts(fontpaths=None, fontext='ttf') else None
    ax[0].set_title('Numeric (5s)', fontsize=30, fontweight='bold', fontname=font_name, pad=20)
    ax[1].set_title('Ours (0.003s)', fontsize=30, fontweight='bold', fontname=font_name, pad=20)

    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')

    ax[0].axis('off')
    ax[1].axis('off')

    plt.text(1, -0.06, '400x400 Grid', ha='right', va='bottom', transform=plt.gca().transAxes)

    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, name + '.pdf'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, name + '.png'), bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    # np.save(os.path.join(save_path, name + '_residual.npy'), (u_pred - u_num).reshape(solver.grid_size, solver.grid_size))

#
# def plot_w(solver, save_path, name, show=False):
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path, exist_ok=True)
#
#     w = solver.w_out.cpu().numpy().reshape(-1)
#     nums = np.arange(w.shape[0])
#
#     w = np.sort(w)
#
#     plt.figure(figsize=(10, 3), dpi=200)
#     plt.scatter(nums, w, s=1)
#     # plt.hist(w, bins='fd')
#     # plt.ylim(0, 15)
#     plt.ylabel('$w_i$')
#     plt.xlabel('$i$')
#     plt.savefig(os.path.join(save_path, name + '_W.pdf'), bbox_inches="tight")
#     plt.savefig(os.path.join(save_path, name + '_W.png'), bbox_inches="tight")
#     if show:
#         plt.show()
#
#
# def plot_H(solver, save_path, name, show=False):
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path, exist_ok=True)
#
#     def flatten_image(image):
#         return image.flatten()
#
#     def sort_images_by_similarity(images, distance_threshold=0.5):
#         flattened_images = np.array([flatten_image(image) for image in images])
#
#         # Compute pairwise distances
#         pairwise_distances = pdist(flattened_images, metric='euclidean')
#
#         # Perform hierarchical clustering
#         linkage_matrix = linkage(pairwise_distances, method='single')
#
#         # Extract clusters
#         clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
#
#         # Sort images by cluster
#         sorted_indices = np.argsort(clusters)
#
#         return sorted_indices
#
#     # Load your images into a list, e.g.:
#     # images = [img1, img2, img3, ...]
#
#     x = solver.x.cpu().numpy().reshape(-1)
#     y = solver.y.cpu().numpy().reshape(-1)
#
#     xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
#     ind = np.lexsort((xy[:, 0], xy[:, 1]))
#
#     x = x[ind]
#     y = y[ind]
#     H = solver.H[ind, :].cpu().numpy().reshape(solver.grid_size - 2, solver.grid_size - 2, -1)
#
#     num_H = H.shape[-1]
#     num_x = int(num_H // 15)
#     num_y = int(num_H / num_x) + 1
#
#     sort = np.argsort(np.mean(np.mean(H, axis=0), axis=0))
#     H = H[:, :, sort]
#
#     images = []
#     for i in range(H.shape[-1]):
#         Hi = H[:, :, i]
#         maximum = np.max(Hi)
#         minimum = np.min(Hi)
#         Hi = (Hi - minimum) / (maximum - minimum)
#         images.append(Hi)
#
#     # Get the sorted indices of the images
#     sorted_indices = sort_images_by_similarity(images)
#     print(sorted_indices)
#     H = H[:, :, sorted_indices]
#
#     fig, axs = plt.subplots(num_y, num_x, figsize=(num_x // 2 + 1, num_y // 2 + 1), dpi=300, sharey='row',
#                             sharex='col')
#
#     for i, ax in enumerate(axs.flat):
#         ax.axis('off')
#         if i < num_H:
#             ax.imshow(images[sorted_indices[i]])
#
#     fig.subplots_adjust(wspace=0.1, hspace=0.1)
#     plt.savefig(os.path.join(save_path, name + '_H.pdf'), bbox_inches="tight")
#     plt.savefig(os.path.join(save_path, name + '_H.png'), bbox_inches="tight")
#     if show:
#         plt.show()
