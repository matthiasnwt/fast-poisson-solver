Welcome to Fast Poisson Solver's documentation!
===============================================



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   solver
   analyze

.. contents:: Table of Contents
   :depth: 2

=============
Installation
=============

This project is available on PyPI and can be installed with pip.

.. code-block:: bash

    pip install fast-poisson-solver

Ensure that you have the latest version of pip; if you're unsure, you can upgrade pip with the following command:

.. code-block:: bash

    pip install --upgrade pip

You might need to use `pip3` instead of `pip`, depending on your system.

===========
Basic Usage
===========

The Fast Poisson Solver is designed to be highly adaptable and flexible. It can accept a variety of input formats, including lists, numpy arrays, and PyTorch tensors. Please note that currently, only 2D domains are supported.

The solver requires `x` and `y` coordinates for both the PDE domain and the boundary region. The order of these coordinates does not matter. For each pair of `x` and `y` coordinates, the PDE domain needs a value for the source function, and the boundary domain needs a value for the boundary condition.

Once the pre-computation for each domain and boundary region is done, it can be used for unlimited different source functions and boundary condition values.

Here is a basic example:

.. code-block:: python

    from fast_poisson_solver import Solver
    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f1, u_bc1)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f2, u_bc2)
    # ...

Please replace x_pde, y_pde, x_bc, y_bc, f, and u_bc with your actual data or variables.
The ... indicates that more solve() method calls can be made with different source functions and boundary conditions.

In the following sections, we will describe the `Solver` class and its methods in more detail.
We will also cover the usage of numeric, plotting, and analyzing methods.

=====================
Solver Class Overview
=====================

The `Solver` class is the core component of the `fast_poisson_solver` package, designed to efficiently solve Poisson's equation with numerous configurations.

.. code-block:: python

    from fast_poisson_solver import Solver
    solver = Solver(device='cuda:0', precision=torch.float32, verbose=False,
                    use_weights=True, compile_model=True, lambdas_pde=None, seed=0)

The `Solver` class is initialized with parameters to specify the computational device ('cpu' or 'cuda'), precision of computation (`torch.float32` or `torch.float64`), verbosity of logs, usage of pre-trained or random weights, compilation of the network model for faster inference, weights for the PDE part in the loss term, and seed for generating random numbers.

The `precompute` method is then used to prepare the data for the solver based on provided coordinates. This step can be performed once for multiple different source functions and boundary condition values.

.. code-block:: python

    solver.precompute(x_pde, y_pde, x_bc, y_bc)

The `solve` method solves the Poisson equation with the provided source function and boundary condition.

.. code-block:: python

    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)

For more complex usage scenarios, such as managing computation resources or integrating with other PyTorch-based workflows, the `Solver` class offers advanced customization options. For instance, the `precompute` method includes `save` and `load` parameters to control the storage and retrieval of precomputed data.

.. code-block:: python

    solver.precompute(x_pde, y_pde, x_bc, y_bc, name='my_precomputation', save=True, load=False)

For a detailed understanding of the `Solver` class, its methods, and their parameters, please refer to the detailed documentation `here <solver.html>`_.
