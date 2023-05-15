Solver
======


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

For a detailed understanding of the `Solver` class, its methods, and their parameters, please refer to the detailed documentation below.

.. autoclass:: fast_poisson_solver.Solver
   :members: precompute, solve
