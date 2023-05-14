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

### Advanced
There are multiple possiblities to 

#### `Solver` Class
The `Solver` class is a key component of the `fast_poisson_solver` package.
This class represents the main solver used for fast Poisson equation solving.

You can initialize as instance of the `Solver` class as follows:

``` python
from fast_poisson_solver import Solver

solver = Solver(device='cuda:0', precision=torch.float32, verbose=False,
                use_weights=True, compile_model=True, lambdas_pde=None, seed=0)
```
Below is a brief explanation of the parameters used in the initialization of the Solver class:

* device (default 'cuda:0'): Specifies the device where the computations will be performed. This should be a valid PyTorch device string such as 'cuda:0' for GPU processing or 'cpu' for CPU processing.
* precision (default torch.float32): This determines the precision of the computation. This should be a valid PyTorch data type such as torch.float32 or torch.float64.
* verbose (default False): A boolean value which, when set to True, enables the printing of detailed logs during computation.
* use_weights (default True): A boolean value which determines whether the network uses pre-trained weights or random weights. If True, the network uses pre-trained weights.
* compile_model (default True): A boolean value which specifies whether the network model is compiled for faster inference. If False, the model won't be compiled.
* lambdas_pde (default None): A list of floats that weight the influence of the PDE part in the loss term. If None, default weight 1e-11 will be used.
* seed (default 0): This parameter sets the seed for generating random numbers, which helps in achieving deterministic results.


## Features



## Contributing

We use SemVer for versioning.


## License


## Contact