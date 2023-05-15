Analyzer
========


To assess the accuracy of your solver's predictions, use the ``analyze`` function.
This function compares your original and predicted source terms, solutions, and boundary conditions, calculating various error metrics for each.
Before using this function, the numerical solution has to be obtained.
See `this section <numeric.html>`_ for details on the numeric solver that is shipped with this package.

Usage of ``analyze`` function is as follows:

.. code-block:: python

    from fast_poisson_solver import Solver, numeric_solve, analyze

    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)
    u_num = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc)

    error_metrics = analyze(f, f_pred, u_pred, u_bc_pred, u_num, u_bc)

The default behavior is to normalize the input arrays before calculating the error metrics, which can be disabled by setting ``normalize=False``.

.. code-block:: python

    error_metrics = analyze(f, f_pred, u_pred, u_bc_pred, u_num, u_bc, normalize=False)

The output is a dictionary, with separate sub-dictionaries for the source term (``f``), solution (``u``), and boundary condition (``bc``), each containing the respective error metrics.

For more on ``analyze`` and its error metrics, see the detailed documentation below.


.. autofunction:: fast_poisson_solver.analyze