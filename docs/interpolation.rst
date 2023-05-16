=============
Interpolation
=============

As the numeric solver does not accept non-grid domains, the package ships with a function that can interpolate between two coordinate sets.

Usage of ``bicubic_interpolate`` function is as follows:

.. code-block:: python

    from fast_poisson_solver import Solver, numeric_solve, bicubic_interpolate

    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)

    u_num_grid = numeric_solve(f_grid, x_pde_grid, y_pde_grid, u_bc_grid, x_bc_grid, y_bc_grid)

    u_num = bicubic_interpolate(x_pde_grid, y_pde_grid, x_bc_grid, y_bc_grid, u_num_grid, x_pde, y_pde, x_bc, y_bc)

This performs a bicubic interpolation from the grid coordinates that the numeric solver needed to the coordinates the Solver takes.#

The function can also interpolate the source function. In this case ``domain=True`` hast to be specified, to do the interpolation only inside the domain.

.. code-block:: python

    bicubic_interpolate(x_pde_grid, y_pde_grid, x_bc_grid, y_bc_grid, f_grid, x_pde, y_pde, x_bc, y_bc, domain=True)

Note: In its current version this function needs the coordinates of the boundary even if domain=True.

.. autofunction:: fast_poisson_solver.bicubic_interpolate