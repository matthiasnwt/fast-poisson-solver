# Fast Poisson Solver
<p float="left">
  <img src="/assets/platzhalter.jpg" height="100" />
  <img src="/assets/platzhalter.jpg" height="100" /> 
</p>

Short description of your project, what it does, the motivation behind it, and its core features.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)
7. [To-Dos](#to-dos)

## Installation

This project is available on PyPI and can be installed with pip:

```bash
pip install fast-poisson-solver
```

## Usage
The code is designed to be very flexible on the input and can take lists, numpy arrays and pytorch tensors as input.
Currently only 2d domains are supported.
The code needs x and y coordinates for the pde domain and x and y coordinates for the boundary region.
The coordinates order does not matter.
For each pair of x and y coordinate the pde domains needs a value for the source function and the boundary domain one for the boundary value.
For each domain and boundary region, the pre-computational has to be done only ones.
Then unlimited different source functions and boundary condition values can be calculated.
### Basic

```python
from fast_poisson_solver import Solver
solver = Solver()
solver.precompute(x_pde, y_pde, x_bc, y_bc)
u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f1, u_bc1)
u_ml, u_ml_pde, u_ml_bc, f_ml, t_ml = solver.solve(f2, u_bc2)
...
```

In the following the class and the two methods are described in more detail.
Also, the usage of the numeric, plotting and analyzing method are described.

### `Solver` Class
The `Solver` class is a key component of the `fast_poisson_solver` package.
This class represents the main solver used for fast Poisson equation solving.

You can initialize as instance of the `Solver` class as follows:

``` python
solver = Solver(device='cuda:0', precision=torch.float32, verbose=False,
                use_weights=True, compile_model=True, lambdas_pde=None, seed=0)
```
Below is a brief explanation of the parameters used in the initialization of the Solver class:
* **device** (default `'cuda'`): Specifies the device where the computations will be performed. This should be a valid PyTorch device string such as `'cuda'` for GPU processing or `'cpu'` for CPU processing.
* **precision** (default `torch.float32`): This determines the precision of the computation. This should be a valid PyTorch data type such as `torch.float32` or `torch.float64`.
* **verbose** (default `False`): A boolean value which, when set to `True`, enables the printing of detailed logs during computation.
* **use_weights** (default `True`): A boolean value which determines whether the network uses pre-trained weights or random weights. If `True`, the network uses pre-trained weights.
* **compile_model** (default `True`): A boolean value which specifies whether the network model is compiled for faster inference. If `False`, the model won't be compiled.
* **lambdas_pde** (default `None`): A list of floats that weight the influence of the PDE part in the loss term. If `None`, default weight `1e-11` will be used.
* **seed** (default `0`): This parameter sets the seed for generating random numbers, which helps in achieving deterministic results.


### `precompute` Method
The `precompute` method is used for precomputing of the data based on the given coordinates.
This method is part of the `Solver` class and can be used as follows:

```python
solver.precompute(x_pde, y_pde, x_bc, y_bc, name=None, save=True, load=True)
```
Here is a brief explanation of the parameters for this method:
* **x_pde**, **y_pde**: These are the coordinates that lie inside the domain and define the behavior of the partial differential equation (PDE). They should be provided as should be provided as a tensor, an array, or a list. Please note that using tensors would lead to the fastest computations.
* **x_bc**, **y_bc**: These are the coordinates of the boundary condition. They should be provided as a tensor, an array, or a list. Please note that using tensors would lead to the fastest computations.
* **name** (default `None`): This is an optional parameter that specifies the name used for saving or loading the precomputed data. If no name is provided, the default name will be used.
* **save** (default `True`): A boolean value that specifies whether the precomputed data should be saved. If `True`, the data will be saved using the provided `name`.
* **load** (default `True`): A boolean value that specifies whether the precomputed data with the provided `name` should be loaded. If `True`, the method will attempt to load the data with the given name.

### `solve` Method
The `solve` method is used to solve the PDE with the provided source function and boundary condition. 
This method is part of the `Solver` class and can be used as follows:
```python
solver.solve(f, bc)
```
Here is a brief explanation of the parameters for this method:
* **f**: This is the source function for the PDE. It should be provided as a tensor, an array, or a list. Please note that using tensors would lead to the fastest computations.
* **bc**: This is the boundary condition for the PDE. It should also be provided as a tensor, an array, or a list. As with the source function, using tensors would result in the fastest computations.

They should be provided as tensors, arrays or lists (Note: tensors are the fastest).
## Features
## Contributing

We use SemVer for versioning.


## License


## Contact

## To-Dos
* Implement a check for cuda and automatically switch to cpu if cuda is not avaiable
