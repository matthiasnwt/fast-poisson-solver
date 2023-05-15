Welcome to Fast Poisson Solver's documentation!
===============================================


The Poisson equation is an integral part of many physical phenomena, yet its computation
is often time-consuming. This module presents an efficient method using
physics-informed neural networks (PINNs) to rapidly solve arbitrary 2D Poisson
problems. Focusing on the 2D Poisson equation, the
method used in this module shows significant speed improvements over the finite difference
method while maintaining high accuracy in various test scenarios.

This module comes with an easy to use method for solving arbitrary 2D Poisson problems.
It also includes a numerical solver and an analyzing function to quantify the results.
For visual inspection, the module offers multiple plotting methods.

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   solver
   numeric
   analyze
   plotting