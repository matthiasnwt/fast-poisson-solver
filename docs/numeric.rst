Numeric
========

To make validating this module for your use case, the module is shiped with a numeric solver that takes the same input format.
The numeric solver is based on Finite Difference Matrix Operators.

The code for the numeric solver is from this `Github repository <https://github.com/zaman13/Poisson-solver-2D>`_ by M.A. Zaman.

See his publication for more details:

Zaman, M.A. "Numerical Solution of the Poisson Equation Using Finite Difference Matrix Operators", Electronics 2022, 11, 2365. https://doi.org/10.3390/electronics11152365

Usage of ``numeric_solve`` function is as follows:

.. code-block:: python

    from fast_poisson_solver import Solver, numeric_solve, analyze

    u_num = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc)

The function accepts lists, numpy arrays, or PyTorch tensors as input data.
It explicitly accepts the same input as the Solver.
However, the coordinates have to form a grid but the order is not important, as the function sorts them anyway.

For more details and further input variables see the full documentation below.

.. autofunction:: fast_poisson_solver.numeric_solve