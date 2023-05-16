Analyzer
========


To assess the accuracy of your solver's predictions, use the ``analyze`` function.
This function compares the predicted source function with the input source function and the predicted boundary condition with the true boundary condition.
If the numeric and predicted solution are provided, they are compared too.

The function calculates the following metrices and returns those as dictionary:

* Source function ``'f'``
    * Mean Square Error ``'MSE'``
    * Root Mean Square Error ``'RMSE'``
    * Mean Absolute Error ``'MAE'``
    * Relative Mean Absolute Error ``'rMAE'``
    * Structural Similarity Index Measure ``'SSIM'``
    * Peak Signal-to-Noise Ratio ``'PSNR'``
    * Coefficient of Determination ``'R2'``
* Boundary Condition ``'bc'``
    * ``'MSE'``
    * ``'RMSE'``
    * ``'MAE'``
    * ``'rMAE'``
* Solution ``'u'``
    * ``'MSE'``
    * ``'RMSE'``
    * ``'MAE'``
    * ``'rMAE'``
    * ``'SSIM'``
    * ``'PSNR'``
    * ``'R2'``


Usage of ``analyze`` function is as follows:

.. code-block:: python

    from fast_poisson_solver import Solver, analyze

    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)

    error_metrics = analyze(f_ml, f, u_ml_bc, u_bc)

The default behavior is to normalize the input arrays before calculating the error metrics, which can be disabled by setting ``normalize=False``.

.. code-block:: python

    error_metrics = analyze(f_pred, f, u_bc_pred, u_bc, normalize=False)

If the solution should also be evaluated, the numerical solution has to be calculted first.

.. code-block:: python

    from fast_poisson_solver import Solver, numeric_solve, analyze

    solver = Solver()
    solver.precompute(x_pde, y_pde, x_bc, y_bc)
    u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f, u_bc)
    u_num = numeric_solve(f, x_pde, y_pde, u_bc, x_bc, y_bc)

    error_metrics = analyze(f_ml, f, u_ml_bc, u_bc)

The output is a dictionary, with separate sub-dictionaries for the source term (``'f'``), solution (``'u'``), and boundary condition (``'bc'``), each containing the respective error metrics.

For more on ``analyze`` and its error metrics, see the detailed documentation below.


.. autofunction:: fast_poisson_solver.analyze