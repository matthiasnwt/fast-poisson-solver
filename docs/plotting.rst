Plotting
========

This module ships with two functions for plotting.
One offers a visual comparison with the numeric solution and one to just display the predicted solution.

Usage of ``plot_comparison`` function for comparing the machine learning and the numeric solution is as follows:

.. code-block:: python

    from fast_poisson_solver import Solver, numeric_solve, plot_comparison

    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)
    u_num = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc)

    plot_comparison(x_pde, y_pde, x_bc, y_bc, u_pred, f, f_pred, u_num)

Usage of ``plot`` function for displaying the machine learning solution is as follows:

.. code-block:: python

    from fast_poisson_solver import Solver, numeric_solve, plot

    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)
    u_num = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc)

    plot(x_pde, y_pde, x_bc, y_bc, u_pred, f, f_pred)

Both function offer the possibility to safe the plot or turning showing the plot off.

See the detailed documentation below for more information's.

.. autofunction:: fast_poisson_solver.plot_comparison
.. autofunction:: fast_poisson_solver.plot